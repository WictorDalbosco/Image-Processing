# SCC251 - Image Processing - 2023/01
# Wictor Dalbosco Silva - 11871027

# Assignment 4 - Mathematical morphology

#Libraries
import numpy as np
import imageio

# Função recursiva para relaizar o Flood Fill numa imagem binaria (.tiff)
def floodFill(img, c, i, j, base, output):
  
  	# Caso base: se o pixel na posição (i, j) não for da mesma cor que a base, significa que já foi pintado anteriormente ou é de uma cor diferente
	if img[i, j] != base:
		return

    # Caso contrário, pintamos o pixel com a nova cor (representada por 1-base)
	img[i, j] = 1 - base
    # Adicionando as coordenadas do pixel ao vetor de saída output.
	output.append((i, j))

	# 4 coordenadas dos vizinhos horizontais e verticais
	floodFill(img, c, i+1, j, base, output)
	floodFill(img, c, i, j+1, base, output)
	floodFill(img, c, i-1, j, base, output)
	floodFill(img, c, i, j-1, base, output)

	# Verificando se é connected-8 neighborhood
	if c == 8:

		# Coordenadas dos vizinhos das diagonais
		floodFill(img, c, i+1, j+1, base, output)
		floodFill(img, c, i-1, j-1, base, output)
		floodFill(img, c, i+1, j-1, base, output)
		floodFill(img, c, i-1, j+1, base, output)
		
		return


def main():
    
    # Vetor de saída
    output = [] 
    
    # Lendo a imagem binária .tiff   
    binaryImg = (imageio.v2.imread(input()) > 127).astype(np.uint8)
    
    # Recebendo as seeds de coordenadas x e y
    seed_X = int(input())
    seed_Y = int(input())
    
    # Conectividade c, (4 ou 8)
    c = int(input())
    
    floodFill(binaryImg, c, seed_X, seed_Y, binaryImg[seed_X,seed_Y], output)
    
    #Ordenando para imprimir, primeiro por i e depois por j
    output.sort(key = lambda x: (x[0], x[1]))
    
    #Aplicando a formatação necessária para output
    for i in range(len(output)):
    	print('(' + str(output[i][0]) + ' ' + str(output[i][1]) + ')', end = ' ')

    
if __name__ == "__main__":
	main()