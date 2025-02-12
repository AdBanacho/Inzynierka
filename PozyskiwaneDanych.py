from PIL import Image
from glob import glob
import os
import numpy as np
px = 9
wymiar=1376*1032

def readminmax(i):
    minmax = open("C:/Users/Adrian/Desktop/inzynierka/WhiteDotRed/RGBminmax.txt", "r")
    for line in minmax:
        if int(line[6:8]) == i:
            rmax = int(line[14:17])
            rmin = int(line[18:21])
            gmax = int(line[22:25])
            gmin = int(line[26:29])
            bmax = int(line[30:33])
            bmin = int(line[34:37])
            minmax.close()
            return rmax, rmin, gmax, gmin, bmax, bmin
           
def WhiteDot():
    way = "C:/Users/Adrian/Desktop/inzynierka/FISH"
    copyG = "C:/Users/Adrian/Desktop/inzynierka/WhiteDotGreen"
    copyR = "C:/Users/Adrian/Desktop/inzynierka/WhiteDotRed"
    os.chdir(way)
    file = glob("FISH (26).tif")
    img = np.empty(20580,dtype=object)
    Dot = np.empty(wymiar,dtype=object)
    #for i in range(len(file)):
    for i in range(1):
        try:
            img[i] = Image.open(file[i])
            w,h = img[i].size
            RGB = img[i].getdata()
            os.chdir(copyR)               # RED
            rmax, rmin, gmax, gmin, bmax, bmin = readminmax(26)
            print(rmax, rmin, gmax, gmin, bmax, bmin)
            for o in range(h*w):
                if (RGB[o][0]<=rmax and RGB[o][0]>=rmin) and (RGB[o][1]<=gmax and RGB[o][1]>=gmin) and (RGB[o][2]<=bmax and RGB[o][2]>=bmin):
                    Dot[o] = (255,255,255)
                else: 
                    Dot[o] = (0,0,0)
            img[i].putdata(Dot)
            img[i].save(file[i])
            
            os.chdir(way)
            
        except IOError: 
                 pass
                 
def Boxw(RGB,o,deepw):
    if RGB[o+deepw][0]>=220 and RGB[o+deepw][1]>=220 and RGB[o+deepw][2]>=220 and deepw < 5:
        deepw+=1
        deepw=Boxw(RGB,o,deepw)
    return deepw                 

def Boxh(RGB,o,deeph,w):
    if RGB[o+deeph*w][0]>=220 and RGB[o+deeph*w][1]>=220 and RGB[o+deeph*w][2]>=220 and deeph < 5:
        deeph+=1
        deeph=Boxh(RGB,o,deeph,w)
    return deeph

def CropBox(file,cornerx,cornery,o,category):
    way = "C:/Users/Adrian/Desktop/inzynierka/FISH"
    copyG = "C:/Users/Adrian/Desktop/inzynierka/GreenBox"
    copyR = "C:/Users/Adrian/Desktop/inzynierka/RedBox"
    os.chdir(way)
    img = np.empty(20580,dtype=object)
    try:
            img = Image.open(file)
            os.chdir(copyG)
            box = (cornerx, cornery, cornerx+px, cornery+px)
            img.crop(box).save("%d_%d " % (category,o) + file)
            
    except IOError: 
                 pass


def BoxBox():
    way = "C:/Users/Adrian/Desktop/inzynierka/RedBox"
    copyG = "C:/Users/Adrian/Desktop/inzynierka/GreenBox"
    copyR = "C:/Users/Adrian/Desktop/inzynierka/RedBox"
    os.chdir(way)
    file = glob("*.tif")
    img = np.empty(20580,dtype=object)
    Box = np.zeros(wymiar,dtype=object)
    for i in range(wymiar):
        Box[i]=(0,0,0)
    for i in range(len(file)):
        try:
            img[i] = Image.open(file[i])
            w,h = img[i].size
            RGB = img[i].getdata()
            os.chdir(copyR)
            for o in range((h-px-3)*(w-px-3)):
                if RGB[o][0]>=220 and RGB[o][1]>=220 and RGB[o][2]>=220:
                     if Box[o][1]<=120:
                        deepw = Boxw(RGB,o,1)
                        deeph = Boxh(RGB,o+int(deepw//2),1,w)
                        center = o+int(deepw//2)+w*int(deeph//2)
                        corner = center-(px//2)-(px//2)*w
                        cornerx = corner%w
                        cornery = corner//w+1
                        #CropBox(file[i],cornerx,cornery,o,2)
                        for j in range(px):
                            for h in range(px):
                                Box[corner+j*w+h] = (0,180,0)
                        Box[center] = (0,255,0)
                else:
                    if Box[o][1]<120 and o%500 == 0:
                        cornerx = o%w
                        cornery = o//w+1
                        CropBox(file[i],cornerx,cornery,o,0)
            img[i].putdata(Box)
            img[i].save(file[i])
            os.chdir(way)
            
        except IOError: 
                 pass
