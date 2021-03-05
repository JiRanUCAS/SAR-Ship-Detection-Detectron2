import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np

def draw_rect(im, true, pred, save_path=None):
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(im[...,::-1])
    plt.axis('off')
    
    for cord in pred:
        x, y, w, h = cord
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor='none', fill=False)
        ax.add_patch(rect)
        
    for cord in true:
        x, y, w, h = cord
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='g', facecolor='none', fill=False)
        ax.add_patch(rect)        
        
    if save_path:
        plt.savefig(f"{save_path}.jpg", bbox_inches='tight')  
    else:
        plt.show()
        
    fig = plt.gcf() #获取当前figure

    plt.close(fig) #关闭传入的 figure 对象    