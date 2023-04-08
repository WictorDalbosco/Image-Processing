# SCC251 - Image Processing - 2023/01
# Wictor Dalbosco Silva - 11871027

# Assignment 1 - Enhancement and Superresolution

#Libraries
import numpy as np
import imageio

# Define função para calular o Root Mean Squared Error (RMSE)
def error(img_modificada, original):
    # Computa erro da imagem processada comparado a imagem de referência
    n, m = img_modificada.shape
    return np.sqrt(np.sum(np.square(np.subtract(original.astype(np.float64), img_modificada.astype(np.float64))))/(n*m))

# Define um histograma de uma imagem
def histogram(image, num_levels):
    
    # cria um histograma vazio com tamanho proporcional ao número de níveis de cinza
    hist = np.zeros(num_levels).astype(int)

    # para todos os níveis de cinza no intervalo
    for gray_level in range(num_levels):
        # soma todas as posições em que o valor de pixel na imagem é igual ao nível de cinza
        pixels_with_gray_level = np.sum(image == gray_level)
        # armazena no array de histograma
        hist[gray_level] = pixels_with_gray_level
            
    return hist

# Abre as 4 imagens de baixa resolução e as armazena num vetor
def open_images(img_filename):
    imgs_low = [0] * 4;
    for i in range(4):
        imgs_low[i] = imageio.v2.imread(f"{img_filename}{i}.png")
        
    return imgs_low

# Seção de Enhancement e Superresolution -------------------------------------
def super_resolution(low_res_imgs):
    
    # Obter dimensões das imagens de entrada
    image_shape = low_res_imgs[0].shape

    # Definir dimensões da imagem de saída (Altura x largura)
    altura = image_shape[0] * 2
    largura = image_shape[1] * 2
    output_shape = (image_shape[0]*2, image_shape[1]*2)

    # Inicializar matriz de imagem vazia para armazenar a imagem de saída
    output_image = np.zeros(output_shape, dtype=np.uint8)
    
    # Iterando pelas linhas
    for row in range(altura):
        if row % 2 == 0: # Linha par
            output_image[row, 0:largura:2] = low_res_imgs[0][row//2] # Completando a linha com os valores 
            output_image[row, 1:largura:2] = low_res_imgs[1][row//2] # alternados entre imagens 1 e 2
        else: # Linha impar
            output_image[row, 0:largura:2] = low_res_imgs[2][(row-1)//2] # Completando a linha com os valores 
            output_image[row, 1:largura:2] = low_res_imgs[3][(row-1)//2] # alternados entre imagens 3 e 4
            #                        ^
            #      vamos de 0 ou 1 ao tamanho da largura da imagem, de 2 em 2
            
    return output_image

# Define função para calcular o histograma cumulativo de uma única imagem
def single_img_cumul_hist(low_res_imgs):
    
    # Imagens equalizadas  
    img_eq = [0] * 4
    
    for i in range(4):   
        
        # Calcula o histograma da imagem
        hist = histogram(low_res_imgs[i], 256)

        # Calcula o histograma cumulativo
        cumul_hist = np.cumsum(hist)

        # Normaliza o histograma cumulativo
        cumul_hist_norm = cumul_hist / cumul_hist.max()

        # Calcula a transformada de equalização de histograma
        eq_hist = (cumul_hist_norm * 255).astype(np.uint8)

        # Aplica a transformada de equalização de histograma na imagem
        img_eq[i] = eq_hist[low_res_imgs[i]]
        
    # Retorna com super resolução
    return super_resolution(img_eq)

def joint_cumul_hist(low_res_imgs):
    
    # Somar as 4 imagens originais em uma (dividir por 4 para o valor estar na média)
    summed_image = np.zeros(low_res_imgs[0].shape) # criar um array com a mesma forma que as imagens
    for img in low_res_imgs:
        summed_image += img
        
    summed_image /= 4
    
    # Aplicar uma single img cumulative histogram para a imagem gerada
    
    # Calcula o histograma da imagem somada
    hist = histogram(summed_image, 256)

    # Calcula o histograma cumulativo
    cumul_hist = np.cumsum(hist)

    # Normaliza o histograma cumulativo
    cumul_hist_norm = cumul_hist / cumul_hist.max()

    # Calcula a transformada de equalização de histograma
    eq_hist = (cumul_hist_norm * 255).astype(np.uint8)
    
    #Aplicar a transformada obtida nas 4 imagens originais
    img_eq = [0] * 4
    
    for i in range(4):
        img_eq[i] = eq_hist[low_res_imgs[i]]
    
    return super_resolution(img_eq) 

# Define função para realizar a correção de gama de 4 imagens
def gamma_correc(low_res_imgs,gamma):
    
    new_imgs = [0] * 4
    
    # Aplicando a correção gama para as 4 imagens de resolução baixa
    for i in range(4):
        new_imgs[i] =  np.floor(255 * ((low_res_imgs[i].astype(np.float64)/255.0)**(1/gamma)))
    
    # Retorna com super resolução  
    return super_resolution(new_imgs)
          
def main():
    
    # Input dos parâmetros do programa
    # basename é o prefixo comum a todas as imagens de baixa resolução
    basename = str(input()).strip()
    # filename é o nome do arquivo de alta resolução para o cálculo do erro
    filename = str(input()).strip()
    # methodID é o identificador do método a ser usado para o processamento de imagem
    methodID = int(input())
    # methodParam é o parâmetro a ser passado para o método escolhido
    methodParam = float(input())
    
    # Carregamento da imagem de alta resolução
    img = imageio.v2.imread(filename)

    # Carregamento das quatro imagens de baixa resolução e armazenamento em um vetor
    imgs_low_res = open_images(basename)

    # Seleção do método de processamento de imagem escolhido
    if methodID == 0:
        # Método de superresolução
        high_img = super_resolution(imgs_low_res)
        
    if methodID == 1:
        # Método de equalização de histograma de imagem única
        high_img = single_img_cumul_hist(imgs_low_res)

    if methodID == 2:
        # Método de histograma Joint Cumulativo
        high_img = joint_cumul_hist(imgs_low_res)

    if methodID == 3:
        # Método de correção gama
        high_img = gamma_correc(imgs_low_res,methodParam)

    # Cálculo do erro RMSE entre a imagem de alta resolução e a imagem processada
    rmse_error = error(high_img, img)

    # Exibição do resultado
    print('%.4f' % rmse_error)
    
if __name__ == "__main__":
    main()
