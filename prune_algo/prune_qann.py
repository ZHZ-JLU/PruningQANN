import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .alps import ALPS_prune, ALPS_prune_after_quant
from .ours import ALPS_prune_ours
# from .data import get_loaders 
from utils import get_loaders
from .ablate import AblateGPT 
from quantize.int_linear import QuantLinear
import logging
import gc
from quantize.utils import set_quant_state
logger = logging.getLogger(__name__)

# def forward_one_layer(layer, inp, attention_mask=None, position_ids=None, past_key_values=None, bs=1, T=1):
#     past_key_values = spike_get_kv_cache(T, past_key_values, bs=bs)
#     outs = layer(inp.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values)
    
#     return outs

# def forward_snn_layer(layer, inp, attention_mask=None, position_ids=None, past_key_values=None, bs=1, T=1):
#     with torch.no_grad():
#         outs = layer(inp, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values)
#     return outs

def find_layers(module, layers=[nn.Linear, QuantLinear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model, args):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        raise ValueError
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

@torch.no_grad()
def draw_loss(dense_model, sparse_model, args, tokenizer, device):
    dense_model.config.use_cache = False 
    sparse_model.config.use_cache = False 
    
    if "llama" in args.model:
        dense_layers = dense_model.model.layers
        sparse_layers = sparse_model.model.layers
        
        dense_model.model.embed_tokens = dense_model.model.embed_tokens.to(device)
        dense_model.model.norm = dense_model.model.norm.to(device)
    elif "opt" in args.model:
        dense_layers = dense_model.model.decoder.layers
        sparse_layers = sparse_model.model.decoder.layers
        
        dense_model.model.decoder.embed_tokens = dense_model.model.decoder.embed_tokens.to(device)
        dense_model.model.decoder.embed_positions = dense_model.model.decoder.embed_positions.to(device)
        if hasattr(dense_model.model.decoder, "project_out") and dense_model.model.decoder.project_out:
            dense_model.model.decoder.project_out = dense_model.model.decoder.project_out.to(device)
        if hasattr(dense_model.model.decoder, "project_in") and dense_model.model.decoder.project_in:
            dense_model.model.decoder.project_in = dense_model.model.decoder.project_in.to(device)
    else:
        raise ValueError

    count = 0 
    total_params = 0
    loss = {}
    weight_loss = {}
    
    dtype = next(iter(dense_model.parameters())).dtype

    dense_inps = torch.zeros(
        (args.nsamples, args.model_seqlen, dense_model.model.config.hidden_size), dtype=dtype, device=device
    )
    dense_outs = torch.zeros_like(dense_inps)
    sparse_outs = torch.zeros_like(dense_inps)
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    dataloader, _ = get_loaders("pile", tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            dense_inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    dense_layers[0] = Catcher(dense_layers[0])
    for batch in dataloader:
        try:
            dense_model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass

    dense_layers[0] = dense_layers[0].module
    torch.cuda.empty_cache()
    sparse_inps = dense_inps.clone()

    if "llama" in args.model:
        dense_model.model.embed_tokens = dense_model.model.embed_tokens.cpu()
        dense_model.model.norm = dense_model.model.norm.cpu()
    elif "opt" in args.model:   
        dense_model.model.decoder.embed_tokens = dense_model.model.decoder.embed_tokens.cpu()
        dense_model.model.decoder.embed_positions = dense_model.model.decoder.embed_positions.cpu()
        if hasattr(dense_model.model.decoder, "project_out") and dense_model.model.decoder.project_out:
            dense_model.model.decoder.project_out = dense_model.model.decoder.project_out.cpu()
        if hasattr(dense_model.model.decoder, "project_in") and dense_model.model.decoder.project_in:
            dense_model.model.decoder.project_in = dense_model.model.decoder.project_in.cpu()
    else:
        raise ValueError
    
    attention_mask = cache['attention_mask']
    if "llama" in args.model:
        position_ids = cache['position_ids']
    
    for i in range(len(dense_layers)):
        dense_layer = dense_layers[i].float().to(device)
        sparse_layer = sparse_layers[i].float().to(device)
        
        dense_inps = dense_inps.float()
        sparse_inps = sparse_inps.float()
        
        for j in range(args.nsamples):
            if "llama" in args.model:
                dense_outs[j] = dense_layer(dense_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                sparse_outs[j] = sparse_layer(sparse_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                dense_outs[j] = dense_layer(dense_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                sparse_outs[j] = sparse_layer(sparse_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
        dense_layers[i] = dense_layer.cpu()
        sparse_layers[i] = sparse_layer.cpu()
        del dense_layer, sparse_layer
        torch.cuda.empty_cache()

        cur_loss = torch.mean( (dense_outs.float() - sparse_outs.float())**2 ).item()
        loss[f"layer_{i}"] = cur_loss
        logger.info(f"layer {i} loss {cur_loss:.6f}"
                    )
        dense_inps, dense_outs = dense_outs, dense_inps
        sparse_inps, sparse_outs = sparse_outs, sparse_inps

    torch.cuda.empty_cache()

    logger.info("Total loss:")
    logger.info(loss)

def prepare_calibration_input(model, dataloader, device, args):
    batch_size = None
    use_cache = model.config.use_cache
    model.config.use_cache = True
    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    dtype = next(iter(model.parameters())).dtype

    model_seqlen = args.model_seqlen
    inps = torch.zeros((args.nsamples, model_seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            B, L, _ = inp.shape  # [1. 4096, 4096]
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model or "Qwen" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(device), past_key_values=spike_get_kv_cache(1, past_key_values, bs=batch[0].shape[0]))
            model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    position_ids = None
    attention_mask = cache['attention_mask']
    if "llama" in args.model or "Qwen" in args.model:
        position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, batch_size

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device, prune_n=0, prune_m=0):
    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    else:
        raise ValueError
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device, prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False
    model_seqlen = args.model_seqlen
    print(f"=====calibration_dataset: {args.cal_dataset}=====")
    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    else:
        raise ValueError
    
    print("loading calibdation data")
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    dataloader, _ = get_loaders(args.cal_dataset, tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, batch_size = prepare_calibration_input(model, dataloader, device, args)
    
    
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(device), outs.to(device)
            attention_mask = attention_mask.to(device)
            if "llama" in args.model or "Qwen" in args.model:
                position_ids = position_ids.to(device)
        else:
            inps, outs = inps.to(device), outs.to(device)
            attention_mask = attention_mask.to(device)
            if "llama" in args.model or "Qwen" in args.model:
                position_ids = position_ids.to(device)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data.to(wrapped_layers[name].scaler_row.device), out.data, None)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            scaler = wrapped_layers[name].scaler_row
            W = subset[name].weight.data.to(wrapped_layers[name].scaler_row.device)
            W_metric = torch.abs(W) * torch.sqrt(scaler.view(1, -1))
            # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                args.use_variant = False
                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    if "out" in name:
                        torch.save(W_mask, f'./wanda/{name}_{i}_before_mask.pt')

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print(f"=====calibration_dataset: {args.cal_dataset}=====")
    dataloader, _ = get_loaders(args.cal_dataset, tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # layers = model.model.layers
    batch_size = None
    model_seqlen = args.model_seqlen
    # model.model.embed_tokens = model.model.embed_tokens.to(device)
    # model.model.norm = model.model.norm.to(device)
    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    else:
        raise ValueError
    
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros(
        (args.nsamples, model_seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model or "Qwen" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass
        
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if "llama" in args.model or "Qwen" in args.model:
        position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(device) #.float()
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {device}")
            inps, outs = inps.to(device), outs.to(device)
            attention_mask = attention_mask.to(device)
            if "llama" in args.model or "Qwen" in args.model:
                position_ids = position_ids.to(device)
        else:
            inps, outs = inps.to(device), outs.to(device)
            attention_mask = attention_mask.to(device)
            if "llama" in args.model or "Qwen" in args.model:
                position_ids = position_ids.to(device)
        
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data.to(gpts[name].H.device), out.data, None)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))


        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
            
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
            
        layers[i] = layer.cpu() #.half()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = True
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_alps(args, model, tokenizer, device, prune_n, prune_m):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # layers = model.model.layers
    batch_size = None
    model_seqlen = args.model_seqlen
    print(f"=====calibration_dataset: {args.cal_dataset}=====")
    dataloader, _ = get_loaders(args.cal_dataset, tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)

    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    else:
        raise ValueError
    # model.model.embed_tokens = model.model.embed_tokens.to(device)
    # model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    inps = torch.zeros(
        (args.nsamples, model_seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model or "Qwen" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "llama" in args.model or "Qwen" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if "llama" in args.model or "Qwen" in args.model:
        position_ids = cache['position_ids']

    # seqlen = model.seqlen

    print('Ready.')

    tot_params = 0
    tot_nnz = 0

    for i in range(len(layers)):
        
        
        layer = layers[i].to(device) #.float()
        full = find_layers(layer)

        attention_mask = attention_mask.to(device)
        if "llama" in args.model or "Qwen" in args.model:
            position_ids = position_ids.to(device)
        sequential = [list(full.keys())]
        scd = {}
        print('----')

        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                scd[name] = ALPS_prune(subset[name], nsamples=args.nsamples, seqlen=model_seqlen, device=device)

            def add_batch(name):
                def tmp(_, inp, out):
                    scd[name].add_batch(inp[0].data, out.data, None)
                return tmp

            handles = []
            
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                
            for j in range(args.nsamples):
                if "llama" in args.model or "Qwen" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    raise ValueError
            
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)

                scd[name].ALPS_admm(sp=args.sparsity_ratio, nm_n=prune_n, nm_m=prune_m, rho=0.1)

                d1 = scd[name].layer.weight.data.shape[0]
                d2 = scd[name].layer.weight.data.shape[1]
                nnz = len( (scd[name].layer.weight.data.abs() > 0).nonzero(as_tuple=True)[0])
                tot_params += d1*d2
                tot_nnz += nnz
                
                scd[name].free()
                
        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
        layers[i] = layer.cpu() #.half()
        del layer
        del scd 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
def mAIHT():
    pass


def prune_ours(args, model, tokenizer, device, prune_n, prune_m):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    if "llama" in args.model or "Qwen" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    else:
        raise ValueError
    
    batch_size = None
    model_seqlen = args.model_seqlen
    print(f"=====calibration_dataset: {args.cal_dataset}=====")
    dataloader, _ = get_loaders(args.cal_dataset, tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)

    # model.model.embed_tokens = model.model.embed_tokens.to(device)
    # model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    inps = torch.zeros(
        (args.nsamples, model_seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model or "Qwen" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    if "llama" in args.model or "Qwen" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if "llama" in args.model or "Qwen" in args.model:
        position_ids = cache['position_ids']

    # seqlen = model.seqlen

    print('Ready.')

    tot_params = 0
    tot_nnz = 0

    for i in range(len(layers)):
        
        
        layer = layers[i].to(device)
        full = find_layers(layer)

        attention_mask = attention_mask.to(device)
        if "llama" in args.model or "Qwen" in args.model:
            position_ids = position_ids.to(device)
            
        sequential = [list(full.keys())]
        scd = {}
        print('----')

        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                scd[name] = ALPS_prune_ours(subset[name], nsamples=args.nsamples, seqlen=model_seqlen, device=device, lamb=args.lamb)

            def add_batch(name):
                def tmp(_, inp, out):
                    # scd[name].add_batch(subset[name].input_dequant.data, out.data, None)
                    scd[name].add_batch(inp[0].data, out.data, None)
                    # if hasattr(scd[name], 'act_quantizer'):
                    #     del subset[name].input_dequant.data 
                return tmp

            handles = []
            
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            set_quant_state(layer, weight_quant=False, act_quant=False)
            for j in range(args.nsamples):
                if "llama" in args.model or "Qwen" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    raise ValueError
                
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)

                # scd[name].ALPS_admm(sp=args.sparsity_ratio, nm_n=prune_n, nm_m=prune_m, rho=0.00001)
                scd[name].ALPS_admm(sp=args.sparsity_ratio, nm_n=prune_n, nm_m=prune_m, rho=args.rho)
                d1 = scd[name].layer.weight.data.shape[0]
                d2 = scd[name].layer.weight.data.shape[1]
                nnz = len( (scd[name].layer.weight.data.abs() > 0).nonzero(as_tuple=True)[0])
                tot_params += d1*d2
                tot_nnz += nnz
                
                scd[name].free()
                
                scd[name].layer.weight = scd[name].layer.weight_quantizer.dequant_weight()
                
        for j in range(args.nsamples):
            if "llama" in args.model or "Qwen" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                raise ValueError
            
        layers[i] = layer.cpu()
        del layer
        del scd 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    
def prune_alps_after_quant(args, model, tokenizer, device, prune_n, prune_m):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    batch_size = None
    model_seqlen = args.model_seqlen

    dataloader, _ = get_loaders("c4", tokenizer, train_size=args.nsamples, val_size=0, seed=args.seed, seqlen=args.model_seqlen)

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    inps = torch.zeros(
        (args.nsamples, model_seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            batch_size = batch[0].shape[0]
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # seqlen = model.seqlen

    print('Ready.')

    tot_params = 0
    tot_nnz = 0

    for i in range(len(layers)):
        
        
        layer = layers[i].to(device)
        full = find_layers(layer)

        attention_mask, position_ids = attention_mask.to(device), position_ids.to(device)
        sequential = [list(full.keys())]
        scd = {}
        print('----')

        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                scd[name] = ALPS_prune_after_quant(subset[name], nsamples=args.nsamples, seqlen=model_seqlen, device=device)

            def add_batch(name):
                def tmp(_, inp, out):
                    scd[name].add_batch(inp[0].data, out.data, None)
                return tmp

            handles = []
            
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            set_quant_state(layer, weight_quant=False, act_quant=True)
            
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                # if "mlp.gate_proj" not in name:
                #     continue

                scd[name].ALPS_admm(sp=args.sparsity_ratio, nm_n=prune_n, nm_m=prune_m, rho=1.0)

                d1 = scd[name].layer.weight.data.shape[0]
                d2 = scd[name].layer.weight.data.shape[1]
                nnz = len( (scd[name].layer.weight.data.abs() > 0).nonzero(as_tuple=True)[0])
                tot_params += d1*d2
                tot_nnz += nnz
                
                scd[name].free()
                
                scd[name].layer.weight = scd[name].layer.weight_quantizer.dequant_weight()
                
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del scd 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()