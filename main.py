from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
from random import randint
from time import time
from sys import argv

"""
Syntaxe de la commande with optional parameters:
python3 main.py (input) (nq) (mq) (factor) (perc)

Optional Parameters:
(input): name of the input file (if no extension, .jpeg by default)
(nq) and (mq): (nq,mq) is the size of each pixel-pannel 
(factor): Size factor increasement for final image
(perc): Randomness rate in the choice of the panel, in percent:
choose the k-th nearest panel with k random in the given percentage of the panels
"""

##################
# Configuration  #
##################
fnamedef = "main"
extdef = "jpeg"
nqdef, mqdef = 10,10
factordef= 3
mixratedef = 0
dirdat, dirin, dirout = "data/", "input/", "output/"
# For debug
savesimp, saveasso, savefinal = 0 ,0 ,0 

################
# Subroutines  #
################
def resize(im,nnew,mnew):
	"""Resize im to have shape (nnew,mnew,3)"""
	n,m,k = im.shape
	nq, mq = n//nnew, m//mnew
	imnew = np.zeros((nnew,mnew,k), dtype=np.uint8)
	for i in range(nnew):
		for j in range(mnew):
			imnew[i,j]=np.uint8(np.round(np.mean(im[nq*i:nq*(i+1),mq*j:mq*(j+1)],axis=(0,1))))
	return imnew

def norm(x,y):
	"""Compute the euclidean distance between two colors x and y, seen as array of R^3"""
	x1, y1 = np.array(x), np.array(y)
	return np.sqrt(np.sum((x-y)**2))

########################################
# Load the parameters in command line  #
########################################
nparam=len(argv)-1
if nparam>=1:
	
	finput = argv[1]
	ff=finput.rsplit('.',1)
	if len(ff)==2:
		fname, fext = ff
	else:
		fname, fext = finput, extdef
		finput = fname+'.'+extdef
else:
	finput = fnamedef+'.'+extdef
	fname, fext = fnamedef, extdef

if nparam>=2:
	nq, mq = map(int,argv[2:4])
else:
	nq, mq = nqdef, mqdef

if nparam>=4:
	factor = int(argv[4])
else:
	factor = factordef

names = listdir("panels/")
nnames = len(names)
if nparam>=5:
	perc = max([0,min([100,int(argv[5])])])
	mixrate = max([0,(nnames*perc)//100-1])
else:
	perc = mixratedef
	mixrate = mixratedef

strparam='_'.join(map(str,[nq,mq,factor,perc]))
strparampan='_'.join(map(str,[nq,mq,factor]))

im = Image.open(dirin+finput)
imnp = np.copy(np.array(im))

n, m, k = imnp.shape
nnew, mnew = n//nq, m//mq

#############################################################
# Resize original image for having as much pixels as panels #
#############################################################
print("Compute imsimp...")
imsimp = resize(imnp,nnew,mnew)
if savesimp:
	np.save(dirdat+"imsimp.npy", imsimp)

#######################################################################
# Resize panels for having size required. Reuse previous computation #
#######################################################################
if path.exists(dirdat+"panels"+strparampan+".npy"):
	print("Load pixpanels...")
	panels = np.load(dirdat+"panels"+strparampan+".npy")
	pixpanels = np.load(dirdat+"pixpanels"+strparampan+".npy")
else:
	print("Compute pixpanels...")
	panels, pixpanels  = [], []
	for s in names:
		p = np.array(Image.open("panels/" + s))
		p = resize(p, factor*nq, factor*mq)
		pixp = np.mean(p,axis=(0,1))
		panels.append(p)
		pixpanels.append(pixp)
	makedirs(dirdat, exist_ok=True)
	np.save(dirdat+"panels"+strparampan+".npy", panels)
	np.save(dirdat+"pixpanels"+strparampan+".npy", pixpanels)

##########################################
# Choose for each pixel, which panel use #
##########################################
print("Compute asso...")
asso = np.array([ [ np.argsort( np.sum((imsimp[i,j]-pixpanels)**2,axis=1) )[randint(0,mixrate)] for j in range(mnew) ] for i in range(nnew) ])

if saveasso:
	np.save(dirdat+"asso.npy", asso)

##############################################
# Compose final image using panels as pixels #
##############################################
print("Compute final image...")
imfinal = np.zeros((factor*n,factor*m,k), dtype=np.uint8)
for i in range(nnew):
	for j in range(mnew):
		imfinal[factor*nq*i:factor*nq*(i+1),factor*mq*j:factor*mq*(j+1)] = panels[asso[i,j]]

if savefinal:
	np.save(dirdat+"imfinal.npy", imfinal)

imfinalPIL = Image.fromarray(np.uint8(imfinal))
imfinalPIL.save(dirout + fname + strparam + '.' + fext)	

