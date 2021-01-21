from yolo.modeling.layers.nn_blocks import ConvBN
from .config_classes import convCFG


def split_converter(lst, i, j=None):
  if j is None:
    return lst.data[:i], lst.data[i:j], lst.data[j:]
  return lst.data[:i], lst.data[i:]


def load_weights(convs, layers):
  # min_key = min(layers.keys())
  # max_key = max(layers.keys())
  keys = sorted(layers.keys())
  for i in keys:  # range(min_key, max_key + 1):
    try:
      cfg = convs.pop(0)
      # print(cfg.c, cfg.filters, layers[i]._filters)
      layers[i].set_weights(cfg.get_weights())
    except BaseException:
      print(f"an error has occured, {layers[i].name}, {i}")


def load_weights_backbone(model, net):
  convs = []
  for layer in net:
    if isinstance(layer, convCFG):
      convs.append(layer)

  layers = dict()
  base_key = 0
  alternate = 0
  for layer in model.layers:
    # non sub module conv blocks
    if isinstance(layer, ConvBN):
      if base_key + alternate not in layers.keys():
        layers[base_key + alternate] = layer
      else:
        base_key += 1
        layers[base_key + alternate] = layer
      # print(base_key + alternate, layer.name)
      base_key += 1
    else:
      #base_key = max(layers.keys())
      for sublayer in layer.submodules:
        if isinstance(sublayer, ConvBN):
          if sublayer.name == "conv_bn":
            key = 0
          else:
            key = int(sublayer.name.split("_")[-1])
          layers[key + base_key] = sublayer
          # print(key + base_key, sublayer.name)
          if key > alternate:
            alternate = key
      #alternate += 1

  load_weights(convs, layers)
  return


def ishead(out_conv, layer):
  # print(out_conv, layer)
  try:
    if layer.filters == out_conv:
      return True
  except BaseException:
    if layer._filters == out_conv:
      return True
  return False


def load_head(model, net, out_conv=255):
  convs = []
  cfg_heads = []
  for layer in net:
    if isinstance(layer, convCFG):
      if not ishead(out_conv, layer):
        convs.append(layer)
      else:
        cfg_heads.append(layer)

  layers = dict()
  heads = dict()
  for layer in model.layers:
    # non sub module conv blocks
    if isinstance(layer, ConvBN):
      if layer.name == "conv_bn":
        key = 0
      else:
        key = int(layer.name.split("_")[-1])

      if ishead(out_conv, layer):
        heads[key] = layer
      else:
        layers[key] = layer
    else:
      for sublayer in layer.submodules:
        if isinstance(sublayer, ConvBN):
          if sublayer.name == "conv_bn":
            key = 0
          else:
            key = int(sublayer.name.split("_")[-1])
          if ishead(out_conv, sublayer):
            heads[key] = sublayer
          else:
            layers[key] = sublayer
          # print(key, sublayer.name)

  load_weights(convs, layers)
  try:
    load_weights(cfg_heads, heads)
  except BaseException:
    print(heads, cfg_heads)
  return cfg_heads


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
