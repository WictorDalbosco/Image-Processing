# SCC251 - Image Processing - 2023/01
# Wictor Dalbosco Silva - 11871027

# Assignment 3 - Image descriptors

#Biliotecas
import numpy as np
import imageio
import scipy

# Para parte de transformar imagem em preto e branco utilizaremos as funções que o preofessor passou em aula
def normalizeMinmax(f, factor):
    fMin = np.min(f)
    fMax = np.max(f)
    f = (f - fMin)/(fMax-fMin)
    return (f*factor)

def luminance(img):
    img = np.array(img, copy=True).astype(float)

    # Computando a conversão
    newImg = np.zeros((img.shape[0], img.shape[1]))
    newImg = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
    newImg = normalizeMinmax(newImg, 255)
    
    return newImg

# Funções para calcular o histograma de gradientes orientados de uma img ----------------------

def computeHog(img):

    # Matrizes de operador Sobel
    wSx = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
    wSy = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    #Realizando a convulucao
    gX = scipy.ndimage.convolve(img, wSx)
    gY = scipy.ndimage.convolve(img, wSy)
    
    #Matrizes para calcular a magnitude e o angulo
    magnitude = np.zeros((256, 256), dtype=np.float64)
    angle = np.zeros((256, 256), dtype=np.float64)

    # Aplicando as fórmulas descritas para computar a magnitude e o angulo
    magnitude = np.hypot(gX, gY)
    somatorio = np.sum(magnitude)
        
    for x in range(256):
        for y in range(256):
            magnitude[x, y] = np.sqrt(gX[x, y] ** 2 + gY[x, y] ** 2) / somatorio
            division = gY[x, y] / gX[x, y]
            angle[x, y] = np.degrees(np.arctan(division) + np.pi/2)

    # Discretizar os angulos em bins
    angleBins = np.digitize(angle, bins=np.arange(0, 190, 20))

    # Montar os histogramas e dar append na lista
    histogram, bin = np.histogram(angleBins.flatten(), bins=np.arange(1,11), weights=magnitude.flatten())

    return histogram


## Funçoes para realizar a predicao usando KNN ------------------------------------------

def euclidianDistance(hist1, hist2):
    return np.sqrt(np.sum((hist1 - hist2)**2))

def predictHuman(testImg, trainImgsWithHumans, trainImgsWithoutHumans, k=3):
  
    # Calcula as distancias para as imagens com humanos (1)
    distancesWithHumans = []
    for i in range(len(trainImgsWithHumans)):
        distance = euclidianDistance(trainImgsWithHumans[i], testImg)
        #Vetor de tupla: distância, e o valor 1, simbolizando q a distancia esta relacionada a imagem com humanos
        distancesWithHumans.append((distance,1)) 

    # Calcula as distancias para as imagens sem humanos (0)
    distancesWithoutHumans = []
    for i in range(len(trainImgsWithoutHumans)):
        distance = euclidianDistance(trainImgsWithoutHumans[i], testImg)
        #Vetor de tupla: distância, e o valor 0, simbolizando q a distancia esta relacionada a imagem com humanos
        distancesWithoutHumans.append((distance,0)) 

    # Vetor de distancias 
    distances = distancesWithHumans + distancesWithoutHumans
    # Ordenar pelo primeiro valor da tupla (distancia em si)
    distances = sorted(distances, key=lambda x: x[0]) 
    
    # Computa os votos, ou seja, segundo valor da tupla, para verificar se há humanos ou nao 
    votes = [distances[i][1] for i in range(k)]
    count0 = 0
    count1 = 0

    for vote in votes:
        if vote == 0:
            count0 += 1
        else:
            count1 += 1
    if count1 > count0:
        return 1
    else:
        return 0

def main():
    
    #Recebendo parâmetros --------------------------------------------------------------
    
    # imgsWithouHumans indica a collection das imagens sem humanos
    fileImgsWithoutHumans = input().split()
    
    # fileImgsWithHumans indica a collection das imagens com humanos
    fileImgsWithHumans = input().split()
    
    # fileImgsWithHumans indica a collection das imagens teste
    fileTestsImgs = input().split()
   
    ## Abrindo as imagens ---------------------------------------------------------------
    imgsWithoutHumans = [] 
    imgsWithHumans = []
    testImgs = []

    # Abrindo as imagens, pode ser que a quantidade varie, logo temos q fazer 3 fors diferentes
    
    # Iimagens com humanos
    for i in range(len(fileImgsWithoutHumans)):
        imgsWithoutHumans.append(imageio.v2.imread(f"./{fileImgsWithoutHumans[i]}"))

    # Imagens sem humanos
    for i in range(len(fileImgsWithHumans)):
        imgsWithHumans.append(imageio.v2.imread(f"./{fileImgsWithHumans[i]}"))

    # Imagens teste
    for i in range(len(fileTestsImgs)):
        testImgs.append(imageio.v2.imread(f"./{fileTestsImgs[i]}"))

    ## Deixando em Preto e Branco ---------------------------------------------------------
    bwImgsWithoutHumans = []
    bwImgsWithHumans = []
    bwTestImgs = []

    for i in range(len(imgsWithoutHumans)):
        bwImgsWithoutHumans.append(luminance(imgsWithoutHumans[i]))

    for i in range(len(imgsWithHumans)):
        bwImgsWithHumans.append(luminance(imgsWithHumans[i]))

    for i in range(len(testImgs)):
        bwTestImgs.append(luminance(testImgs[i]))

    # Calculando o hog de todas as imagens --------------------------------------------------    
    hogWithoutHumans = []
    hogWithHumans = []
    hogTestImgs = []

    # Passando pelas imagens, pode ser que a quantidade varie, logo temos q fazer 3 fors diferentes
    for i in range(len(bwImgsWithoutHumans)):
        hogWithoutHumans.append(computeHog(bwImgsWithoutHumans[i]))

    for i in range(len(bwImgsWithHumans)):
        hogWithHumans.append(computeHog(bwImgsWithHumans[i]))

    for i in range(len(bwTestImgs)):
        hogTestImgs.append(computeHog(bwTestImgs[i]))
        
    # Realizando as predições com knn ------------------------------------------------------------------
    for i in range(len(hogTestImgs)):
        print(predictHuman(hogTestImgs[i], hogWithHumans, hogWithoutHumans), end=' ')
        
        
if __name__ == "__main__":
    main()