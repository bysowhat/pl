import matplotlib.pyplot as plt 

def show_2dimg(img):
    img = img.permute(1, 2, 0)
    img = img.cpu().detach().numpy()
    plt.imshow(img) 
    plt.show()