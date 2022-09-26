import numpy as np
import cv2
import math
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

img = cv2.imread( "../Resource/Jenny2.bmp", -1 )

nr, nc = img.shape[:2]

F = fft2(img)  #離散傅立葉轉換(複數陣列)
F = np.fft.fftshift( F )

G = F.copy()

for x in range( nr ) :
    for y in range( nc ) :
        G[x,y] = math.log(1+abs( F[x,y] ),10)

g = np.zeros( [nr,nc] )
g = np.uint8(np.clip(G,0,255))

G2 = F.copy()

for x in range( nr ) :
    for y in range( nc ) :
        G2[x,y] = math.atan2( F[x,y].imag, F[x,y].real )

g2 = np.zeros( [nr,nc] )

g2 = np.uint8(np.clip(G2,0,255))

#cv2.normalize(g, g, 0, 255, cv2.NORM_MINMAX)
#cv2.normalize(g2, g2, 0, 255, cv2.NORM_MINMAX)
#cv2.imwrite("Jenny_log.bmp",g)
#cv2.imwrite("Jenny_phase.bmp",g2)
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(132), plt.imshow(g, 'gray'), plt.title('log Fourier spectrum')
plt.axis('off')
plt.subplot(133),plt.imshow(g2, 'gray'), plt.title('phase spectrum')
plt.axis('off')
plt.show()
cv2.destroyAllWindows()