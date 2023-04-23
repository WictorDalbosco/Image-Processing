# SCC251 - Image Processing - 2023/01
# Wictor Dalbosco Silva - 11871027

# Assignment 2 - Fourier Transform

#Libraries
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Define função para calular o Root Mean Squared Error (RMSE)
def error(img_modificada, original):
    
    # Computa erro da imagem processada comparado a imagem de referência
    n, m = img_modificada.shape
    
    return np.sqrt(np.sum(np.square(np.subtract(original.astype(np.float64), img_modificada.astype(np.float64))))/(n*m))

#Aplica o filtro na imagem e normaliza 
def applyFilter(image,filter):
    
    filteredImage = np.fft.ifft2(np.fft.ifftshift(image * filter))
    filteredImage = ((filteredImage - np.min(filteredImage)) / (np.max(filteredImage) - np.min(filteredImage))) * 255
    filteredImage = np.abs(filteredImage).clip(0,255).astype(np.uint8)
    
    return filteredImage

def calcDist(u,P,v,Q):
    return np.sqrt((u-P/2)**2+(v-Q/2)**2)

# Filtros ------------------------------------------------

def idealLowPass(originalImage,radius):
    
    P,Q = originalImage.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(originalImage))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(P):
            
            dist = calcDist(u,P,v,Q)
            if dist <= radius:
                filter[u,v] = 1
            elif dist > radius:
                filter[u,v] = 0

    filteredImage = applyFilter(freqDomain,filter)
    
    return filteredImage
        

def idealHighPass(originalImage, radius):
   
    P,Q = originalImage.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(originalImage))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            
            dist = calcDist(u,P,v,Q)
            if dist <= radius:
                filter[u,v] = 0
            elif dist > radius:
                filter[u,v] = 1

    filteredImage = applyFilter(freqDomain,filter)
    
    return filteredImage

def idealBandPass(originalImage, radius1, radius2):
    
    P,Q = originalImage.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(originalImage))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            
            dist = calcDist(u,P,v,Q)
            if dist <= radius1 and dist >= radius2:
                filter[u,v] = 1
            else:
                filter[u,v] = 0

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage

def laplacianHighPass(img):
    
    P,Q = img.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(img))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            filter[u,v]= 4*(np.pi**2)*((u-P/2)**2+(v-Q/2)**2)

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage

def gaussianLowPass(originalImage,sigma1,sigma2):
    
    P,Q = originalImage.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(originalImage))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    
    for u in range(P):
        for v in range(Q):
            x = (((u-P/2)**2/(2*sigma1**2))+((v-Q/2)**2/(2*sigma2**2)))
            filter[u,v] = np.exp(-x)

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage

def butterworthLowPass(originalImage,D0,n):
    
    P,Q = originalImage.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(originalImage))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            dist = calcDist(u,P,v,Q)
            filter[u,v] = 1/(1+(dist/D0)**(2*n))

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage

def butterworthHighPass(img,D0,n):
    
    P,Q = img.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(img))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            dist = calcDist(u,P,v,Q)
            filter[u,v] = 1-1/(1+(dist/D0)**(2*n))

    filteredImage = applyFilter(freqDomain,filter)
    
    return filteredImage

def butterworthBandReject(img,D0,D1,n1,n2):
    
    P,Q = img.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(img))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            dist = calcDist(u,P,v,Q)
            filter[u,v] = 1-1/(1+(dist/D1)**(2*n2))+1/(1+(dist/D0)**(2*n1))

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage

def butterworthBandPass(img,D0,D1,n1,n2):
    
    P,Q = img.shape
    freqDomain = np.fft.fftshift(np.fft.fft2(img))
    filter = np.zeros((P,Q), dtype=np.float32)
    
    for u in range(P):
        for v in range(Q):
            dist = calcDist(u,P,v,Q)
            filter[u,v] = 1-1/(1+(dist/D0)**(2*n1))+ 1- 1/(1+(dist/D1)**(2*n2))

    filteredImage = applyFilter(freqDomain,filter)

    return filteredImage
    
def main():
    
    # Input dos parâmetros do programa
    # inputImage indica onde está a imagem de entrada
    inputImageName = str(input()).strip()
    # expectedImage é o nome do arquivo esperado para o cálculo do erro
    expectedImageName = str(input()).strip()
    # filterIndex é o identificador do filtro a ser aplicado
    filterIndex = int(input())
    
    # Carregamento da imagem deentrada
    inputImage = imageio.v2.imread(f"./{inputImageName}")

    # Carregamento da imagem para comparação
    expectedImage = imageio.v2.imread(f"./{expectedImageName}")

    # Seleção do método de processamento de imagem escolhido
    
    if filterIndex == 0: # Ideal Low-Pass
        
        radius = float(input().rstrip())
        filteredImage = idealLowPass(inputImage,radius)
        
    elif filterIndex == 1: # Ideal High-Pass
        
        radius = float(input().rstrip())
        filteredImage = idealHighPass(inputImage,radius)

    elif filterIndex == 2: # Ideal Band-Pass
        
        radius1 = float(input().rstrip())
        radius2 = float(input().rstrip())
        filteredImage = idealBandPass(inputImage,radius1,radius2)

    elif filterIndex == 3: # Laplacian High-Pass
        
        filteredImage = laplacianHighPass(inputImage)
        
    elif filterIndex == 4:# Gaussian Low-Pass
        
        sigma1 = float(input().rstrip())
        sigma2 = float(input().rstrip())
        filteredImage = gaussianLowPass(inputImage,sigma1,sigma2)
        
    elif filterIndex == 5: # Butterworth Low-Pass
        
        D0 = float(input().rstrip())
        n = float(input().rstrip())
        filteredImage = butterworthLowPass(inputImage,D0,n)
        
    elif filterIndex == 6: # Butterworth High-Pass
        
        D0 = float(input().rstrip())
        n = int(input().rstrip())
        filteredImage = butterworthHighPass(inputImage,D0,n)
    
    elif filterIndex == 7: #Butterwoth Band Reject
        
        D0 = float(input().rstrip())
        D1 = float(input().rstrip())
        n1 = int(input().rstrip())
        n2 = int(input().rstrip())
        filteredImage = butterworthBandReject(inputImage,D0,D1,n1,n2)
    
    elif filterIndex == 8: # Butterworth Band Pass
        
        D0 = float(input().rstrip())
        D1 = float(input().rstrip())
        n1 = int(input().rstrip())
        n2 = int(input().rstrip())
        filteredImage = butterworthBandPass(inputImage,D0,D1,n1,n2)

    # Cálculo do erro RMSE entre a imagem de alta resolução e a imagem processada
    rmse_error = error(filteredImage, expectedImage)
        
    # Exibição do resultado
    print('%.4f' % rmse_error)
    
if __name__ == "__main__":
    main()
