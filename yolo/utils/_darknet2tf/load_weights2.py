from yolo.modeling.layers.nn_blocks import ConvBN, DarkRouteProcess, PathAggregationBlock, CSPRoute, CSPConnect, SAM
from .config_classes import convCFG, samCFG
import numpy as np


def split_converter(lst, i, j=None):
  if j is None:
    return lst.data[:i], lst.data[i:j], lst.data[j:]
  return lst.data[:i], lst.data[i:]


def load_weight(cfg, layer):
  weights = cfg.get_weights()
  if len(layer.get_weights()) == len(weights):
    layer.set_weights(weights)
  else:
    # need to match the batch norm
    weights.append(np.zeros_like(weights[-2]))
    weights.append(np.zeros_like(weights[-2]))
    weights.append(np.zeros([]))
    weights.append(np.zeros([]))
    layer.set_weights(weights)


def load_weights(convs, layers):
  # min_key = min(layers.keys())
  # max_key = max(layers.keys())
  keys = sorted(layers.keys())
  #print (layers)

  unloaded = []
  unloaded_convs = []
  for i in keys:  # range(min_key, max_key + 1):

    try:
      cfg = convs.pop(0)
      print(layers[i].name, cfg)
      #print(cfg.c, cfg.filters, layers[i]._filters)
      weights = cfg.get_weights()
      if len(layers[i].get_weights()) == len(weights):
        layers[i].set_weights(weights)
      else:
        # need to match the batch norm
        weights.append(np.zeros_like(weights[-2]))
        weights.append(np.zeros_like(weights[-2]))
        weights.append(np.zeros([]))
        weights.append(np.zeros([]))
        layers[i].set_weights(weights)
    except BaseException as e:
      unloaded_convs.append(cfg)
      unloaded.append(layers[i])
      print(f"an error has occured, {layers[i].name}, {i}, {e}")
  return unloaded, unloaded_convs


def load_weights_backbone(model, net):
  convs = []
  for layer in net:
    if isinstance(layer, convCFG):
      convs.append(layer)

  layers = dict()
  key = 0
  for layer in model.layers:
    # non sub module conv blocks
    # print(layer.name)
    if isinstance(layer, ConvBN):
      layers[key] = layer
      key += 1
    elif "residual_down" in layer.name:
      temp = []
      for sublayer in layer.submodules:
        if isinstance(sublayer, ConvBN):
          #print(sublayer.name, key)
          temp.append(sublayer)

      a = [temp[-1]] + temp[:-1]

      for layeri in a:
        layers[key] = layeri
        #print(layeri.name, key)
        key += 1

    else:
      for sublayer in layer.submodules:
        if isinstance(sublayer, ConvBN):
          #print(sublayer.name, key)
          layers[key] = sublayer
          key += 1

  load_weights(convs, layers)
  # sys.exit()
  return


def load_weights_fpn(model, net, csp=False):
  convs = []
  sam = False
  for layer in net:
    if isinstance(layer, convCFG):
      convs.append(layer)

  layers = dict()
  base_key = 0
  alternate = 0
  for layer in model.submodules:
    # print(layer.name)
    # # non sub module conv blocks
    if isinstance(layer, ConvBN):
      if layer.name == "conv_bn":
        key = 0
      else:
        key = int(layer.name.split("_")[-1])
      layers[key + base_key] = layer
      if key > alternate:
        alternate = key
      alternate += 1
  u, v = load_weights(convs, layers)
  if csp:
    reorg_csp_convs_fpn(u, v)
  return


def load_weights_pan(model, net, csp=False, out_conv=255):
  convs = []
  cfg_heads = []
  sam = 0
  for layer in net:
    if isinstance(layer, convCFG):
      if not ishead(out_conv, layer):
        convs.append(layer)
      else:
        cfg_heads.append(layer)

    if sam > 0:
      convs[-2], convs[-1] = convs[-1], convs[-2]
      sam += 1
      if sam == 3:
        sam = 0

    if isinstance(layer, samCFG):
      sam += 1

  layers = dict()
  key = 0
  base_key = 0
  alternate = 0
  for layer in model.submodules:
    if isinstance(layer, ConvBN):
      if layer.name == "conv_bn":
        key = 0
      else:
        key = int(layer.name.split("_")[-1])
      layers[key + base_key] = layer
      if key > alternate:
        alternate = key
      alternate += 1
  u, v = load_weights(convs, layers)
  if csp:
    reorg_csp_convs_pan(u, v)
  return cfg_heads


def load_weights_decoder(model, net, csp=False):
  layers = dict()
  base_key = 0
  alternate = 0
  if not csp:
    for layer in model.layers:
      # non sub module conv blocks
      print(layer.name)
      if "input" not in layer.name and "fpn" in layer.name:
        load_weights_fpn(layer, net[0], csp=csp)
      elif "input" not in layer.name and "pan" in layer.name:
        out_convs = load_weights_pan(layer, net[1], csp=csp)
    return out_convs
  else:
    return load_csp(net, model)


def deconstruct_route_process(mod):
  if isinstance(mod, DarkRouteProcess):
    dark_convs = []
    print(mod)
    for a in mod.layers:
      if isinstance(a, CSPRoute):
        for b in a.submodules:
          if isinstance(b, ConvBN):
            dark_convs.append(b)
            print("rout conv")
      if isinstance(a, CSPConnect):
        for b in a.submodules:
          if isinstance(b, ConvBN):
            dark_convs.append(b)
            print("connect conv")
      if isinstance(a, SAM):
        for b in a.submodules:
          if isinstance(b, ConvBN):
            dark_convs.append(b)
      if isinstance(a, ConvBN):
        dark_convs.append(a)
        print("conv")
    return dark_convs
  return None


def deconstruct_path_agg(mod):
  if isinstance(mod, PathAggregationBlock):
    print(mod)
    path_convs = []
    for a in mod.submodules:
      if isinstance(a, ConvBN):
        path_convs.append(a)
        print("path conv")
    return path_convs
  return None


def load_csp(net, model, out_conv=255):
  cfg_heads = []
  convs = []

  net_ = []
  for set_in in net:
    net_ += set_in

  net = net_
  for layer in net:
    if isinstance(layer, convCFG):
      if not ishead(out_conv, layer):
        convs.append(layer)
      else:
        cfg_heads.append(layer)

  for layer in model.layers:
    # if isinstance(mod, DarkRouteProcess):
    if "input" not in layer.name and "fpn" in layer.name:
      route_convs = []
      merges = []
      for mod in layer.submodules:
        print(mod.name)
        dark_convs = deconstruct_route_process(mod)
        if dark_convs is not None:
          route_convs.append(dark_convs)
        path_convs = deconstruct_path_agg(mod)
        if path_convs is not None:
          merges.append(path_convs)
      blocks = []
      for i in range(len(route_convs)):
        blocks.extend(route_convs[i])
        try:
          blocks.extend(merges[i])
        except:
          pass
      print(len(blocks), len(convs))

      blocks = blocks[9:] + blocks[0:9]
      for layer in blocks:
        cfg = convs.pop(0)
        load_weight(cfg, layer)
        print(layer.name, layer._filters, layer._kernel_size, cfg)
    if "input" not in layer.name and "pan" in layer.name:
      route_convs = []
      merges = []
      for mod in layer.submodules:
        print(mod.name)
        dark_convs = deconstruct_route_process(mod)
        if dark_convs is not None:
          route_convs.append(dark_convs)
        path_convs = deconstruct_path_agg(mod)
        if path_convs is not None:
          merges.append(path_convs)
      blocks = []
      for i in range(len(route_convs)):
        blocks.extend(route_convs[i])
        try:
          blocks.extend(merges[i])
        except:
          pass
      print(len(blocks), len(convs))

      for layer in blocks:
        cfg = convs.pop(0)
        load_weight(cfg, layer)
        print(layer.name, layer._filters, layer._kernel_size, cfg)

  return cfg_heads


def ishead(out_conv, layer):
  # print(out_conv, layer)
  try:
    if layer.filters == out_conv:
      return True
  except BaseException:
    if layer._filters == out_conv:
      return True
  return False


def load_weights_prediction_layers(convs, model):
  # print(convs)
  try:
    i = 0
    for sublayer in model.submodules:
      if ("conv_bn" in sublayer.name):
        # print(sublayer, convs[i])
        sublayer.set_weights(convs[i].get_weights())
        i += 1
  except BaseException:
    i = len(convs) - 1
    for sublayer in model.submodules:
      if ("conv_bn" in sublayer.name):
        # print(sublayer, convs[i])
        sublayer.set_weights(convs[i].get_weights())
        i -= 1
  return


def load_weights_v4head(model, net, remap):
  convs = []
  for layer in net:
    if isinstance(layer, convCFG):
      convs.append(layer)

  layers = dict()
  base_key = 0
  for layer in model.layers:
    if isinstance(layer, ConvBN):
      if layer.name == "conv_bn":
        key = 0
      else:
        key = int(layer.name.split("_")[-1])
      layers[key] = layer
      base_key += 1
      # print(base_key, layer.name)
    else:
      for sublayer in layer.submodules:
        if isinstance(sublayer, ConvBN):
          if sublayer.name == "conv_bn":
            key = 0 + base_key
          else:
            key = int(sublayer.name.split("_")[-1]) + base_key
          layers[key] = sublayer
          # print(key, sublayer.name)
