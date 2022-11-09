# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:37:12 2022

@author: PC
"""

#%%
import numpy as np
import cv2
# img = np.array(im2)
def get_contrast(img):
    import numpy as np
    from scipy.stats import mode
    
    
    pc = np.array(img)
    pc = (pc).astype(np.uint8)
    pc = cv2.cvtColor(pc, cv2.COLOR_BGR2RGB)
    pc = cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY)
    std = np.std(pc)
    c=  min(mode(pc)[0][0])
    u = pc - c
    u = u**2
    a = np.sum(u)
    a = a / (len(pc)*len(pc[0]))
    a = np.sqrt(a)
    b = np.mean(pc)
    c = min(mode(pc)[0][0])
    return a, b ,c, std

#%%

from PIL import Image,ImageEnhance

# path2 = path1 + flist2[i]
# im2 = Image.open(path2)
# en = ImageEnhance.Brightness(im2).enhance(0.1+oo)

# self.progressBar.setValue(20)
# QtWidgets.QApplication.processEvents()
# self.label_status2.setText('Processing ... Please wait')
# self.label_status2.repaint()
# print('(adjusting brightness and constrast automatically...)')
# nn = []
# t = []
# s = []
# l = []
#brightness (mean)
oo = 0
for i in range(0, 50):
#Display image  
    print(i, oo)
    path2 = 'K:\\SynologyDrive\\study\\dy\\박하늬선생님_1103\\L-dopa\\C1.1L.JPG'
    im = Image.open(path2)
    en = ImageEnhance.Brightness(im).enhance(0.1+oo)
    if (253 > get_contrast(en)[1] > 241):
        break
    else:
        oo += 0.1

# self.progressBar.setValue(60)
# QtWidgets.QApplication.processEvents()
# # contrast (sig)
ii = 0
for i in range(50):
    print(i, ii)
    enh = ImageEnhance.Contrast(en).enhance(0.2+ii)
    if 36 < get_contrast(enh)[3] < 43:
        break
    elif get_contrast(enh)[3] > 43:
        break
    else:
        ii += 0.1
        
print('done')

# self.progressBar.setValue(100)
# QtWidgets.QApplication.processEvents()
# time.sleep(1)
# self.progressBar.setValue(0)

ad_img = np.array(enh)    
e = cv2.cvtColor(ad_img,cv2.COLOR_BGR2RGB)



#%%
import os
import matplotlib.image as img

path1 = 'K:\\SynologyDrive\\study\\dy\\박하늬선생님_1103\\L-dopa\\'
flist = os.listdir(path1); flist2=[]
for i in range(len(flist)):
     if os.path.splitext(flist[i])[1] == '.JPG':
         flist2.append(flist[i])

for i in range(len(flist2)):
    im = img.imread(path1 + flist2[i])
import matplotlib.pyplot as plt
plt.imshow(im)

plt.figsave()
#%% img load 후 contrast / brightness 조절




# im (조절전) -> im_fix (조절후)
#%% keras model load

lnn = 7

# import tensorflow_addons as tfa
# from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras import layers, models
# from tensorflow.keras.layers import BatchNormalization, Dropout
# metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

def model_setup(xs=None, ys=None, lr=1e-4, lnn=7):
    from tensorflow.keras.optimizers import Adam
    model = models.Sequential()
    model.add(layers.Conv2D(2**lnn, (4, 4), activation='relu', input_shape=xs))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**lnn, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**lnn, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    lnn2 = lnn-1
    lnn3 = lnn-2
    
    model.add(layers.Flatten())
    model.add(layers.Dense(2**lnn2, activation='relu' ))
    model.add(layers.Dense(2**lnn2, activation='relu' ))
    model.add(layers.Dense(2**lnn2, activation='relu' ))
    model.add(layers.Dense(2**lnn3, activation='relu' ))
    model.add(layers.Dense(2**lnn3, activation='relu' ))
    model.add(layers.Dense(2**lnn3, activation='relu' ))
    model.add(layers.Dense(2**lnn3, activation='sigmoid') )
    model.add(layers.Dense(ys, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr, decay=1e-3, beta_1=0.9, beta_2=0.999), \
                  loss='categorical_crossentropy') 
    
    return model

SIZE_HF = 14
model = model_setup(xs=(SIZE_HF*2,SIZE_HF*2,3), ys=(2), lnn=lnn)
print(model.summary())

# weight_path = 'cv_0_subject_A3_1L_total_final.h5'
# weight_path = 'totalset_total_final.h5'
weight_path = 'merge_20221017.h5'
# weight_path = 'cv_0_total_final.h5'
model.load_weights(weight_path)

#%%

im = img.imread(path1 + 'C1.1L.JPG')
im = img.imread(path1 + 'C1.1L_modify.JPG')

#%%

#%% # 2. test data prep
from tqdm import tqdm
import pickle

#   import numpy as np
#   t_im = np.array(im); roiinfo=None

def ms_testprep(t_im=None, roiinfo=None, filename=None):
    import numpy as np
    im = np.array(t_im)
    im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, 3)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = im
    
    if roiinfo is None:
        rowmin, colmin = 0, 0
        rowmax = im.shape[0]
        colmax = im.shape[1]
        polygon = None
    if not(roiinfo is None):
        from shapely.geometry import Point
        pass

    yhat_save = []
    z_save = []
    divnum = 100
    forlist = list(range(rowmin, rowmax))
    div = int(len(forlist)/divnum)
    #   div_i = 0
    for div_i in range(divnum):
        print('div', div_i)
        if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
        elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
    
        X_total_te = []
        Z_total_te = []
        for row in tqdm(forlist_div):
            for col in range(colmin, colmax):
                if not(polygon is None):
                    code = Point(col,row)
                    if code.within(polygon):
                        crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                        if crop.shape == (SIZE_HF*2, SIZE_HF*2, 3):
                            for ch in range(3):
                                crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                            X_total_te.append(np.array(crop))
                            Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
                else:
                    crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                    if crop.shape == (SIZE_HF*2, SIZE_HF*2, 3):
                        for ch in range(3):
                            crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                        X_total_te.append(np.array(crop))
                        Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
            
        X_total_te = np.array(X_total_te)
        print(np.mean(np.std(X_total_te,0)))
            
        if len(X_total_te) > 0: # polygon ROI 때문에 필요함
            X_total_te = np.array(X_total_te)
            yhat = model.predict(X_total_te, verbose=1, batch_size = 2**6)
            yhat_save += list(yhat[:,1])
            z_save += Z_total_te
                
    z_save = np.array(z_save)
    msdict = {'yhat_save': yhat_save, 'z_save': z_save}

    if filename is None: filename = 'noname.pickle'
    psave = os.getcwd() + '\\' + filename
    if not(os.path.isfile(psave)):
        with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
            print(psave, '저장되었습니다.')

ms_testprep(t_im=None, roiinfo=None)



