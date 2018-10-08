from __future__     import division
from torch.autograd import Variable

import torch
import torch.nn as tnn
import torch.nn.functional as tfunc
import numpy as np

def parseCfg(cfgFile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file  = open(cfgFile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty,
                                         #   implies it is storing values of previous block.
                blocks.append(block)     # Add it the blocks list
                block = {}               # Re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def createModeules(blocks, numChan=3):
    moduleList    = tnn.ModuleList()
    netInfo       = blocks[0]
    prevFilters   = numChan
    outputFilters = []

    for index, x in enumerate(blocks[1:]):
        module = tnn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # if it's a convolutional layer
        if (x["type"] == "convolutional"):
            # get the info about the layer
            activation = x["activation"]
            try:
                batchNorm = int(x["batch_normalize"])
                bias = False
            except:
                batchNorm = 0
                bias = True

            filters    = int(x["filters"])
            padding    = int(x["pad"])
            kernelSize = int(x["size"])
            stride     = int(x["stride"])

            if    padding: pad = (kernelSize - 1) // 2 # assuming odd kernel size?
            else: pad = 0

            # add the convolutional layer
            conv = tnn.Conv2d(prevFilters, filters, kernelSize, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batchNorm:
                bn = tnn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = tnn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer, we use nearest neighbor
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = tnn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = outputFilters[index + start] + outputFilters[index + end]
            else:
                filters= outputFilters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prevFilters = filters
        outputFilters.append(filters)

    return (net_info, module_list)