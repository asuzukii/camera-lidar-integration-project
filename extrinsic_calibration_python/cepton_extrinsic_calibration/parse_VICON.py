###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help='The VICON capture file to be parsed')
parser.add_argument('-m', '--mode', type=str, required=True)

args = parser.parse_args()
# GM VICON Parser

def parse_data(file : str):
    dict = {}

    with open(file) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('Marker') and "Target" in line:
            if line.split(" ")[-1] == "True":
                continue
            clean_line = line[11:-6]
            
            marker_name = clean_line.split("(")[0][:-1]
            marker_coord = clean_line.split("(")[1][:-1]

            if not marker_name in dict.keys():
                print("({})".format(marker_coord))
                dict[marker_name] = [[float(value.strip()) for value in marker_coord.split(',')]]
            else:
                dict[marker_name].append([float(value.strip()) for value in marker_coord.split(',')])
    
    for key in dict:
        dict[key] = np.array(dict[key])
        dict[key] = np.mean(dict[key], 0).tolist()

    return dict

def compensate_marker_height(TL,TR,BL,BR):
    """
    Used to compensate for the height of Vicon markers, based on what center detection is from Vicon system.
    Inputs shall be in vehicle coordinates.
    Outputs shall be in vehicle coordinates.
    """
    
    TL = np.array(TL)
    TR = np.array(TR)
    BL = np.array(BL)
    BR = np.array(BR)

    markerHeight = 10 # 10mm for marker height
    YVector = BL - TL
    XVector = TR - TL
   
    X = np.copy(XVector)
    Y = np.copy(YVector)
    
    X /= np.linalg.norm(X)
    Y /= np.linalg.norm(Y)
    
    planeVector = np.cross(X,Y)
    planeVector /= np.linalg.norm(planeVector)

    planeVector = planeVector * markerHeight

    if planeVector[1] < 0:
        planeVector *= -1
    print(planeVector)
 
    TL_Compensate = TL + planeVector
    TR_Compensate = TR + planeVector
    BL_Compensate = BL + planeVector
    BR_Compensate = BR + planeVector
 
    return TL_Compensate.tolist(),TR_Compensate.tolist(),BL_Compensate.tolist(),BR_Compensate.tolist()

if __name__ == "__main__":
    markers = parse_data(args.file)

    if args.mode == "train":
        wanted_names = ['TargetB_left', 'TargetA_left', 'TargetC', 'TargetA_right', 'TargetB_right']
    elif args.mode == "validate":
        wanted_names = ['TargetD_left', 'TargetC', 'TargetD_right']

    for name in wanted_names:
        
        print("=========<{}>=========".format(name))
        correspondence = []
        coords = []
        for key in markers.keys():
            if key.startswith(name + '_'):
                x = int(key.split('_')[-2])
                z = int(key.split('_')[-1])

                coords.append(markers[key])
                correspondence.append([x, z])

        comp_coord = compensate_marker_height(coords[0], coords[1], coords[2], coords[3])

        print("{},".format(comp_coord[0]))
        print("{},".format(comp_coord[1]))
        print("{},".format(comp_coord[2]))
        print("{}".format(comp_coord[3]))

        print("{},".format(correspondence[0]))
        print("{},".format(correspondence[1]))
        print("{},".format(correspondence[2]))
        print("{}".format(correspondence[3]))
        
