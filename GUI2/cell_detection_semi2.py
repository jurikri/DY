 # -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:07:08 2022

@author: PC
"""

import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random, pickle, cv2
from shapely.geometry import Point, Polygon
import cv2
from PIL import Image,ImageFilter, ImageEnhance, ImageOps
from tqdm import tqdm
# from PIL import Image,ImageEnhance
import os
import matplotlib.image as img
import pandas as pd
import gc
import time

import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction

#%% rawdata_dict gen

mainpath = 'C:\\SynologyDrive\\study\\dy\\image_data\\processed_imgs\\'
mainpath_common = 'C:\\SynologyDrive\\study\\dy\\image_data\\'
psave = mainpath + 'data_ms.pickle'

with open(psave, 'rb') as file:
    dictionary = pickle.load(file)
keylist = list(dictionary.keys())

#%% """HYPERPARAMETERS"""

# NCT = 50 # INTENSITY 조정값

SIZE_HF = 14 # crop size: 29 x 29 

LNN = 7
CHNUM = 3
BATCH_SIZE = 2**6

lnn = LNN
chnum = CHNUM
batch_size = BATCH_SIZE

# SIZE = (SIZE_HF*2)+1
#%%
def get_F1(threshold=None, contour_thr=None,\
           yhat_save=None, z_save=None, t_im=None, positive_indexs=None, polygon=None):

    import numpy as np
    # import cv2
    
    height = t_im.shape[0]
    width = t_im.shape[1]
    
    noback_img = np.zeros((height, width, 3), np.uint8)
    noback_img[:]= [255,255,0]
    
    # test_img = np.zeros((height, width))

    vix = np.where(np.array(yhat_save) > threshold)[0]
    for i in vix:
        row = z_save[i][0]
        col = z_save[i][1]
        noback_img[row,col] = [0,0,255]
        # test_img[row,col] = 1
        
    # plt.imshow(test_img)

    img_color = noback_img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_save = []

    pink = [] # predicted cell 중앙 좌표
    los = [] # size filter 에서 살아남는 contours 
    
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        if M['m00'] >= contour_thr and  M['m00'] <= 500: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            code = Point(cx,cy)
            if not(polygon is None):
                if code.within(polygon):
                    pink.append((cx,cy))
                    los.append(contours[i])
                    contours_save.append(contours[i])
            else:
                pink.append((cx,cy))
                los.append(contours[i])
                
    def dot_expand(height=None, width=None, dots=None):
        white2 = np.zeros((height,width,3))*np.nan
        white2[:,:] = [255,255,255]
        boxsize = 10
        for z in range(len(dots)):
            row = dots[z][1]
            col = dots[z][0]
            white2[np.max([row-boxsize, 0]) : np.min([row+boxsize, height]), \
                   np.max([col-boxsize, 0]) : np.min([col+boxsize, width])] = 0
        return white2

    dots = list(pink)
    white2 = dot_expand(height=height, width=width, dots=dots)
    predict_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        predict_area.append((w[1][i],w[0][i]))
    # plt.imshow(white2)

    co = []
    for i in range(len(positive_indexs)):
        row = positive_indexs[i][0]
        col = positive_indexs[i][1]
        code = Point(col, row)
        if not(polygon is None):
            if code.within(polygon):
                co.append((int(col),int(row)))
        else: co.append((int(col),int(row)))
        
    dots = list(co)
    white2 = dot_expand(height=height, width=width, dots=dots)
    cell_positive_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        cell_positive_area.append((w[1][i],w[0][i]))
        
    if True:
            # TN
        dots = list(co)
        ansbox_p = dot_expand(height=height, width=width, dots=dots)
        ansbox = np.array(ansbox_p[:,:,0])
        predbox = np.array(white2[:,:,0])
        predbox2 = np.zeros(predbox.shape)
        predbox2[np.where(predbox==0)] = 255
        
        # plt.imshow(predbox2)
        # plt.imshow(ansbox)
        # plt.imshow(ansbox - predbox2)
        # ansbox3 = ((ansbox - predbox2) == 255)
        # plt.imshow(ansbox3)
        # TN = np.sum(ansbox3)
    
        Cell_n = len(co)
        # Predict_n = len(pink)
        
        FN = len(list(set(co) - set(predict_area)))
        TP = Cell_n - FN
        
        FP = len(list(set(pink) - set(cell_positive_area)))
        
        if np.min([FN, TP, FP]) < 0:
            print(threshold, contour_thr)
            import sys; sys.exit()
    
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2 * precision * recall / (precision + recall)
        # acc = (TP + TN) / (TP + TN + FP + FN)
    except: F1_score = 0; #  acc = 0
    
    # plt.imshow(predbox2)
    # plt.imshow(ansbox)
    
    msdict = {'los': los, 'co': co, 'tp': TP, 'fp': FP, 'fn': FN, \
              'predbox2': predbox2, 'ansbox': ansbox, 'pink': pink, 'contours_save': contours_save}
    
    return F1_score, msdict

#%% keras setup
import tensorflow_addons as tfa
metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

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
                  loss='categorical_crossentropy', metrics=metric) 
    
    return model

model = model_setup(xs=(29,29,3), ys=(2), lnn=7)
print(model.summary())


#%%
epand_num = 6
def positive_expand(row=None, col=None, epand_num=epand_num):
    rowcol_list = []
    for rowi in range(-epand_num, epand_num+1):
        for coli in range(-epand_num, epand_num+1):
            if ((rowi**2 + coli**2))**0.5 < epand_num + 0.1:
                rowcol_list.append([row+rowi, col+coli])
    return rowcol_list

rowcol_list = positive_expand(row=100, col=100, epand_num=epand_num)
    
def msXYZgen(im=None, marker_x=None, marker_y=None):
    
    d = epand_num + 0.1; gap=[]; d2 = int(d+1)
    for row in range(-d2,d2):
        for col in range(-d2,d2):
            distance = np.sqrt(row**2 + col**2)
            if distance < d:
                gap.append((row,col))
                
    d = 30; gap2=[]
    for row in range(-d,d):
        for col in range(-d,d):
            distance = np.sqrt(row**2 + col**2)
            if distance <= d:
                gap2.append((row,col))

    def gen_total_index(polygon=None, rectangle_roi_dict=None):
        from shapely.geometry import Point
        
        rowmin = rectangle_roi_dict['rowmin']
        rowmax = rectangle_roi_dict['rowmax']
        colmin = rectangle_roi_dict['colmin']
        colmax = rectangle_roi_dict['colmax']

        total_index = []
        if not polygon is None:
            for row in range(rowmin, rowmax):
                for col in range(colmin, colmax):
                    code = Point(col,row)
                    if code.within(polygon):
                        total_index.append((row,col))
        else:
            for row in range(rowmin, rowmax):
                for col in range(colmin, colmax):
                    total_index.append((row,col))
                    
        return total_index
    
    X, Y, Z = [], [], []
    
    polygon = None

    # padding img gen & 좌표변경
    im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, 3)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = im

    # plt.imshow(im)
    # plt.imshow(im_padding/np.mean(im_padding))
    
    marker_y2 = marker_y + SIZE_HF
    marker_x2 = marker_x + SIZE_HF
    
    width = im_padding.shape[1]
    height = im_padding.shape[0]

    # marker 기준 rectangle ROI 생성
    # market 최 외곽 기준 +- 50 pix로 ROI 가져감
    rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
    rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
    colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
    colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
    rectangle_roi_dict = {'rowmin': rowmin, 'rowmax': rowmax, 'colmin': colmin, 'colmax': colmax}
      
    total_index = gen_total_index(polygon=polygon, rectangle_roi_dict=rectangle_roi_dict)
    
    # positive label
    
    positive_index = [] 
    i = 0
    for i in range(len(marker_x)):
        row = marker_y2[i]
        col = marker_x2[i]
        
        rowcol_list = positive_expand(row=row,col=col)
        for ei in range(len(rowcol_list)):
            row2 = rowcol_list[ei][0]
            col2 = rowcol_list[ei][1]
        
            crop = np.array(im_padding[row2-SIZE_HF:row2+SIZE_HF, col2-SIZE_HF:col2+SIZE_HF, :])
            if crop.shape == (SIZE_HF*2, SIZE_HF*2, 3):
                for ch in range(3):
                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                    # plt.figure(); plt.imshow(crop[:,:,ch])
                    
                X.append(np.array(crop))
                Y.append([0,1])
                Z.append(list([n, row, col]))
        positive_index.append((row, col))
            
    # plt.imshow(X[50])
    # plt.imshow(crop)
        
    gap_index = []
    for n1 in range(len(positive_index)):
        for n2 in range(len(gap)):
            row = positive_index[n1][0] + gap[n2][0]
            col = positive_index[n1][1] + gap[n2][1]
            if (row > SIZE_HF and row < height-SIZE_HF) and (col > SIZE_HF and col < width-SIZE_HF):
                gap_index.append((row,col))
            
            
    gap_index2_pre = []
    for n1 in range(len(positive_index)):
        for n2 in range(len(gap2)):
            row = positive_index[n1][0] + gap2[n2][0]
            col = positive_index[n1][1] + gap2[n2][1]
            if (row > SIZE_HF and row < height-SIZE_HF) and (col > SIZE_HF and col < width-SIZE_HF):
                gap_index2_pre.append((row,col))
            
    eix0 = np.where(np.sum((np.array(gap_index2_pre) < 0), axis=1))[0]
    eix1 = np.where(np.sum((np.array(gap_index2_pre)[:0] > height), axis=1))[0]
    eix2 = np.where(np.sum((np.array(gap_index2_pre)[:1] > width), axis=1))[0]
    eix = set(list(eix0) + list(eix1) + list(eix2))
    tix = list(range(len(gap_index2_pre)))
    tix = set(tix) - set(eix)
    gap_index2 = []
    for jj in tix:
        gap_index2.append(gap_index2_pre[jj])
         
    negative_index = list(set(total_index) - set(positive_index) - set(gap_index))
    negative_index2 = list(set(gap_index2) - set(positive_index) - set(gap_index))
    negative_n = list(range(len(negative_index)))
    negative_n2 = list(range(len(negative_index2)))
    
    # negative crop
    # negative_cnt = 0
    # occupied = np.transpose(np.array([marker_x, marker_y]))
    # negative_n/len(positive_index)
    
    # epochs = 1000000000
    pnum = np.sum(np.array(Y), axis=0)[1]
    
    rix = random.sample(negative_n, int(pnum/5))
    for j in rix:
        row = negative_index[j][0]
        col = negative_index[j][1]
        
        crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
        if crop.shape == (SIZE_HF*2, SIZE_HF*2, 3):
            for ch in range(3):
                crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
            X.append(np.array(crop))
            Y.append([1,0])
            Z.append(list([n, row, col]))

    rix = random.sample(negative_n2, int(len(negative_n2)/5))
    for j in rix:
        row = negative_index2[j][0]
        col = negative_index2[j][1]
        
        crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
        if crop.shape == (SIZE_HF*2, SIZE_HF*2, 3):
            for ch in range(3):
                crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
            X.append(np.array(crop))
            Y.append([1,0])
            Z.append(list([n, row, col]))

    if False: # 확인용인듯
        positive = np.where(np.logical_and(np.array(Z)[:,0]==n, np.array(Y)[:,1]==1))[0]
        negative = np.where(np.logical_and(np.array(Z)[:,0]==n, np.array(Y)[:,0]==1))[0]
        
        allo = np.ones((height,width))
        br = 1
        for y in positive:
            allo[np.max([0, Z[y][1]-br]):np.min([height, Z[y][1]+br]), \
                 np.max([0, Z[y][2]-br]):np.min([width, Z[y][2]+br])] = 2
                
        plt.figure()
        plt.imshow(allo, cmap='binary')
        
        br = 1
        for y in negative:
            allo[np.max([0, Z[y][1]-br]):np.min([height, Z[y][1]+br]), \
                 np.max([0, Z[y][2]-br]):np.min([width, Z[y][2]+br])] = 0

        left = np.where(np.sum(allo==0, axis=1)>0)[0][0]
        right = np.where(np.sum(allo==0, axis=1)>0)[0][-1]
        top = np.where(np.sum(allo==0, axis=0)>0)[0][0]
        bottom = np.where(np.sum(allo==0, axis=0)>0)[0][-1]

        plt.figure()
        plt.imshow(allo[top:bottom,left:right], cmap='binary')
        # plt.title(n)
        
    print(np.mean(np.array(Y), axis=0))
        
    msdict = {'X':X, 'Y':Y, 'Z':Z}
    return msdict

#%% XYZ gen
if False:
    mainpath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\'
    msFunction.createFolder(mainpath)
    
    n_num = 0
    for n_num in tqdm(range(len(keylist))):
        n = keylist[n_num]
        psave = mainpath + 'data_52_ms_XYZ_' + str(n) + '.pickle'
        
        if not(os.path.isfile(psave)) or True:
            marker_x = dictionary[n]['marker_x']
            marker_y = dictionary[n]['marker_y']
            
            rlist = random.sample(list(range(len(marker_x))), 10)
            marker_x = marker_x[rlist]
            marker_y = marker_y[rlist]
            
            im = dictionary[n]['imread']
            # im = np.mean(im, axis=2)
            msdict = msXYZgen(im=im, marker_x=marker_x, marker_y=marker_y)
            msdict['rlist'] = rlist
            with open(psave, 'wb') as file:
                pickle.dump(msdict, file)
                print(psave, '저장되었습니다.')


#%% dataset1 only
mainpath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\'
xyz_loadpath = mainpath
xyz_loadpath2 = 'F:'  + xyz_loadpath[2:]
cvlistsavepath = xyz_loadpath + 'cvlist2.pickle'

if not(os.path.isfile(cvlistsavepath)):
    xyz_loadpath = mainpath
    flist = os.listdir(xyz_loadpath)
    nlist = []
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.pickle' and flist[i][:7] == 'data_52':
            nlist.append(flist[i])
    nlist = list(set(nlist))
    
    rlist = list(range(len(nlist))); 
    random.seed(0); random.shuffle(rlist)    
    nlist = np.array(nlist)[rlist]
    cvnum = 20
    divnum = len(rlist) / cvnum
    cvlist = []
    for cv in range(cvnum):
        cvlist.append(list(range(int(round(cv*divnum)), int(round((cv+1)*divnum)))))
    print(cvlist)
    
    cvlist_msid = []
    for i in range(cvnum):
        cvlist_msid.append(nlist[cvlist[i]])
        
    msdict = {'cvlist_msid': cvlist_msid, 'tnlist': nlist}
    with open(cvlistsavepath, 'wb') as f:  # Python 3: open(..., 'rb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(cvlistsavepath, '저장되었습니다.')

with open(cvlistsavepath, 'rb') as file:
    msdict = pickle.load(file)
    cvlist_msid = msdict['cvlist_msid']
    tnlist = msdict['tnlist']
    
#%%

cv = 0
lnn = 7
repeat = 1
chnum = 3
batch_size = 2**6

weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)
print(weight_savepath)
#%% model training
if False: # 전략상 total dataset으로 바로감
    for cv in range(len(cvlist_msid)):
    # for cv in range(0, len(cvlist)):
        # 1. weight
        print('cv', cv)
        weight_savename = 'cv_' + str(cv) + '_total_final.h5'
        final_weightsave = weight_savepath + weight_savename
    
        if not(os.path.isfile(final_weightsave)) or False:
            telist = cvlist_msid[cv]
            trlist = list(set(tnlist) - set(telist))
            
            model = model_setup(xs=(28, 28, 3), ys=2, lnn=lnn)
            if cv==0: print(model.summary())
            
            resetsw = True
            for i in range(len(trlist)):
                start = time.time()
                gc.collect(); tf.keras.backend.clear_session()
                psave = xyz_loadpath + trlist[i]
                with open(psave, 'rb') as file:
                    msdict = pickle.load(file)
                    
                if resetsw:
                    Xtr = np.array(msdict['X'])
                    Ytr = np.array(msdict['Y'])
                    resetsw = False
                else:
                    Xtr = np.concatenate((Xtr, np.array(msdict['X'])), axis=0)
                    Ytr = np.concatenate((Ytr, np.array(msdict['Y'])), axis=0)
                print('i', i, Xtr.shape)
                
                if (Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]) > 470400000 or i == len(trlist)-1:
                    resetsw = True
                    hist = model.fit(Xtr, Ytr, epochs=4, verbose=1, batch_size = 2**6)
           
            model.save_weights(final_weightsave)
            gc.collect()
            tf.keras.backend.clear_session()
            
            #
mainpath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\'; msFunction.createFolder(mainpath)
weight_savepath = mainpath + 'weightsave_7\\'; msFunction.createFolder(weight_savepath)
weight_savename = 'total_total_final_1.h5'
final_weightsave = weight_savepath + weight_savename
# lnn = 7
if not(os.path.isfile(final_weightsave)):
    
    model = model_setup(xs=(28, 28, CHNUM), ys=2, lnn=LNN)
    print(model.summary())
    resetsw = True
    for i in range(len(tnlist)):
        start = time.time()
        gc.collect(); tf.keras.backend.clear_session()
        psave = xyz_loadpath + tnlist[i]
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            
        if resetsw:
            Xtr = np.array(msdict['X'])
            Ytr = np.array(msdict['Y'])
            resetsw = False
        else:
            Xtr = np.concatenate((Xtr, np.array(msdict['X'])), axis=0)
            Ytr = np.concatenate((Ytr, np.array(msdict['Y'])), axis=0)
        print('i', i, Xtr.shape)
        
        if (Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]) > 470400000 or i == len(tnlist)-1:
            resetsw = True
            hist = model.fit(Xtr, Ytr, epochs=4, verbose=1, batch_size = 2**6)
   
    model.save_weights(final_weightsave)
    gc.collect()
    tf.keras.backend.clear_session()


#%% # 2. test data prep

mainpath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\'; msFunction.createFolder(mainpath)
weight_savepath = mainpath + 'weightsave_7\\'; msFunction.createFolder(weight_savepath)
final_weightsave = weight_savepath + 'total_total_final_1.h5'

model = model_setup(xs=(28, 28, CHNUM), ys=2, lnn=LNN)
model.load_weights(final_weightsave)
print(model.summary())

for i in range(len(tnlist)):
    gc.collect(); tf.keras.backend.clear_session()
    msid = tnlist[i][15:-7]
    
    test_image_no = msid
    print(test_image_no)
    psave = weight_savepath + 'sample_n_' + test_image_no + '.pickle'

    # width = dictionary[test_image_no]['width']
    # height = dictionary[test_image_no]['length']
    
    t_im = dictionary[test_image_no]['imread']
    # t_im = np.mean(t_im, axis=2)
    marker_x = dictionary[test_image_no]['marker_x']
    marker_y = dictionary[test_image_no]['marker_y']
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    polygon = dictionary[test_image_no]['polygon']
    points = dictionary[test_image_no]['points']

    im = np.array(t_im)
    im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, 3)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = im
    marker_y2 = marker_y + SIZE_HF
    marker_x2 = marker_x + SIZE_HF
    width = im_padding.shape[1]
    height = im_padding.shape[0]

    # 2. test data
    if not(os.path.isfile(psave)):
        import time; start = time.time()
        print('prep test data', cv)
        model = model_setup(xs=(28, 28, CHNUM), ys=2, lnn=lnn)
        model.load_weights(final_weightsave)
        
        rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
        rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
        colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
        colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
        
        testarea = []
        for row in range(rowmin, rowmax):
            for col in range(colmin, colmax):
                code = Point(col,row)
                if code.within(polygon):
                    testarea.append([row, col])

        yhat_save = []
        z_save = []
        
        divnum = 10
        forlist = list(range(len(testarea)))
        div = int(len(forlist)/divnum)

        for div_i in range(divnum):
            print('div', div_i)
            if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
            elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
        
            X_total_te = []
            Z_total_te = []
            for i in forlist_div:
                row, col = testarea[i][0], testarea[i][1]

                crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                for ch in range(3):
                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                X_total_te.append(np.array(crop))
                Z_total_te.append([row-SIZE_HF, col-SIZE_HF])

            if len(X_total_te) > 0: # polygon ROI 때문에 필요함
                X_total_te = np.array(X_total_te)
                yhat = model.predict(X_total_te, verbose=1, batch_size = 2**6)
                yhat_save += list(yhat[:,1])
                z_save += Z_total_te
                    
        z_save = np.array(z_save)
        msdict = {'yhat_save': yhat_save, 'z_save': z_save}
        plt.figure(); plt.imshow(im_padding)
      
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
            
        if not(os.path.isfile(psave)) or False:
            with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                print(psave, '저장되었습니다.')

#%% predict to label

def find_inner_coordinate(contours=None):
    # mssave = []   
    rows = np.array(contours)[:,0][:,0]
    cols = np.array(contours)[:,0][:,1]
    
    s1, e1 = np.min(rows), np.max(rows) + 1
    s2, e2 = np.min(cols), np.max(cols) + 1
    
    inside_points = [] # 윤곽선 안에 있는 좌표들을 저장할 리스트
    for row in range(s1, e1):
        for col in range(s2, e2):
            point = (row,col) # x,y 순서로 좌표 생성
            dist = cv2.pointPolygonTest(contours, point, False) # 점과 다각형 사이의 거리와 위치 구하기
            if dist == 1: # 점이 다각형 안에 있으면
                inside_points.append(point) # 리스트에 추가하기
                
    return inside_points

nlist_for = list(tnlist)
mssave2 = []

path3 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\semilabel\\'; msFunction.createFolder(path3) # new label save path
# path3 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post1\\predict\\'; msFunction.createFolder(path3)
weight_savepath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\weightsave_7\\' # exsit predict file path

n_num = 0

for n_num in range(len(keylist)):
    msid = keylist[n_num]
    psavepath = path3 + msid + '_semilabel.pickle'
    if not(os.path.isfile(psavepath)):
        # idnum = int(id_list[i][1]) #  's210331_3L'
        
        # yhat_save, z_save
        psave = weight_savepath + 'sample_n_' + msid + '.pickle'
        # psave2 = weight_savepath + 'F1_parameters_' + msid + '.pickle'
        
        c1 = os.path.isfile(psave)
        
        if c1:
            print(msid)
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                yhat_save = msdict['yhat_save']
                z_save = msdict['z_save']
            
            threshold = 0.5 #result[2]
            contour_thr = 120 # result[3]

            t_im = dictionary[msid]['imread']
            marker_x = dictionary[msid]['marker_x']
            marker_y = dictionary[msid]['marker_y']
            positive_indexs = np.transpose(np.array([marker_y, marker_x]))
            polygon = dictionary[msid]['polygon']
            points = dictionary[msid]['points']
            
            F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                       yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
                
            contours_save = msdict['contours_save']
            print(msid, F1_score)
             
            msdict = {'msid': msid, 'contours_save': contours_save, 'pink': msdict['pink']}
            with open(psavepath, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                print(psavepath, '저장되었습니다.')

#%%

# predict map 만들고, Unet용 XY gen
path3 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\semilabel\\'
path4 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\semilabel\\unetXY\\';
msFunction.createFolder(path4)

# XYgenforUnet
i = 0
for i in range(len(tnlist)):
    msid = tnlist[i][15:-7]
    test_image_no = msid
    print(test_image_no)
    pload_contours_save =  path3 + msid + '_semilabel.pickle'
    psave_unetXY = path4 + msid + '_m_unet_XY.pickle'
    
    with open(pload_contours_save, 'rb') as file:
        msdict = pickle.load(file)
        contours_save = msdict['contours_save']
    
    t_im = dictionary[msid]['imread']
    # marker_x = dictionary[test_image_no]['marker_x']
    # marker_y = dictionary[test_image_no]['marker_y']
    # positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    # polygon = dictionary[test_image_no]['polygon']
    
    tim = np.array(t_im)
    rows = tim.shape[0]
    cols = tim.shape[1]
    allo_tim = np.zeros((rows, cols))
    # THR = 0.5

    for roin in range(len(contours_save)):
        forlist = find_inner_coordinate(contours=contours_save[roin])
        for pix in range(len(forlist)):
            row, col = forlist[pix][1], forlist[pix][0]
            allo_tim[row, col] = 1
    # plt.imshow(allo_tim)
    
    # padding

    # tim = np.array(tim)
    # rows = tim.shape[0]
    # cols = tim.shape[1]
    bins = int(128/2)
    cs = 128
    
    row_for = range(0, rows-cs, bins)
    col_for = range(0, cols-cs, bins)

    Xunet, Yunet, Zunet = [], [], []
    for row in row_for:
        for col in col_for:
            xtmp = tim[row:row+cs, col:col+cs, :]
            xnmr = xtmp/255

            Xunet.append(xnmr)
            Yunet.append(allo_tim[row:row+cs, col:col+cs])
            Zunet.append([msid, row,row+cs, col,col+cs])

    if False: # for vis
        for n in range(100):
            if np.mean(Yunet[n]) > 0:
                plt.figure(); plt.imshow(Xunet[n])
                plt.figure(); plt.imshow(Yunet[n])
                plt.title(str(n))

    Xunet, Yunet, Zunet = np.array(Xunet), np.array(Yunet), np.array(Zunet)
    msdict = {'Xunet':Xunet, 'Yunet':Yunet, 'Zunet':Zunet}

    if not(os.path.isfile(psave_unetXY)) or True:
        with open(psave_unetXY, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
            print(psave_unetXY, '저장되었습니다.')

#%% Unet

# down-stack
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# up-stack
from tensorflow_examples.models.pix2pix import pix2pix
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int, output_bias=None):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])
    
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same', bias_initializer=output_bias)  #64x64 -> 128x128
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

path4 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\semilabel\\unetXY\\';

def model_unet_setup(output_bias=None):
    tf.config.run_functions_eagerly(True)
    model_unet = unet_model(output_channels=2, output_bias=output_bias)
    model_unet.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model_unet

#%%
repeat = 5
resetsw = True; startsw = True; mssave_hist = []
for e in range(repeat):
    for i in range(len(tnlist)):
        gc.collect(); tf.keras.backend.clear_session()
        msid = keylist[i]
        psave_unetXY = path4 + msid + '_m_unet_XY.pickle'
        
        with open(psave_unetXY, 'rb') as file:
            msdict = pickle.load(file)
            
        if resetsw:
            Xtr = np.array(msdict['Xunet'])
            Ytr = np.array(msdict['Yunet'])
            Ztr = np.array(msdict['Zunet'])
            resetsw = False
        else:
            Xtr = np.concatenate((Xtr, np.array(msdict['Xunet'])), axis=0)
            Ytr = np.concatenate((Ytr, np.array(msdict['Yunet'])), axis=0)
            Ztr = np.concatenate((Ztr, np.array(msdict['Zunet'])), axis=0)
        print('i', i, Xtr.shape)
           
        if False: # for vis
            for n in range(100):
                if np.mean(Yunet[n]) > 0:
                    plt.figure(); plt.imshow(Xunet[n])
                    plt.figure(); plt.imshow(Yunet[n])
                    plt.title(str(n))
    
        if (Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]) > 470400000 or i == len(tnlist)-1:
            if startsw and True:
                startsw = False
                initial_bias = np.log([np.sum(Ytr==1)/np.sum(Ytr==0)])
                model_unet = model_unet_setup(output_bias=initial_bias)
    
            resetsw = True
            # hist = model.fit(Xtr, Ytr, epochs=4, verbose=1, batch_size = 2**6)
            hist = model_unet.fit(Xtr, Ytr, epochs=20, batch_size = 2**6)
            # epoch 100까지 계속 떨어짐, 수치는 낮음 5e-04
            mssave_hist.append(hist.history['loss'])
        
final_weightsave = path4 + 'total_total_final_unet.h5'
model_unet.save_weights(final_weightsave)
gc.collect()
tf.keras.backend.clear_session()
    
    #%%
model_unet.load_weights(final_weightsave)

keys = list(set(np.array(Ztr)[:,0]))
msid = keys[1]
vix = np.where(np.array(Ztr)[:,0]==msid)[0]
yhat = model_unet.predict(Xtr[vix])

allo2 = np.zeros((1944, 2592)) * np.nan
allo3 = np.zeros((1944, 2592)) * np.nan

for i in range(len(yhat)):
    ixs = np.array(Ztr[vix][i][1:], dtype=int)
    allo2[ixs[0]: ixs[1], ixs[2]:ixs[3]] = yhat[i][:,:,1]
    allo3[ixs[0]: ixs[1], ixs[2]:ixs[3]] = Ytr[vix][i][:,:]
plt.imshow(allo2)
plt.imshow(allo3)


np.mean(np.abs(Ytr - (yhat[:,:,:,1] > 0)))

if False:
    
    vix = np.where(np.mean(np.mean(Ytr, axis=1), axis=1) > 0.03)[0]
    j = vix[0]
    for j in vix:
        yhat = model_unet.predict(Xtr[[j]])
        plt.figure(); plt.imshow(yhat[0,:,:,1])
        plt.figure(); plt.imshow(Ytr[j])
        plt.title(str(j))
            
#%%

def get_F1_Unet(threshold=None, contour_thr=None,\
           allo=None, t_im=None, positive_indexs=None, polygon=None):

    import numpy as np
    # import cv2
    
    height = t_im.shape[0]
    width = t_im.shape[1]
    
    noback_img = np.zeros((height, width, 3), np.uint8)
    noback_img[:]= [255,255,0]
    
    # test_img = np.zeros((height, width))
    vix = np.where(np.array(allo) > threshold)
    for i in range(len(vix[0])):
        row = vix[0][i]
        col = vix[1][i]
        noback_img[row,col] = [0,0,255]
        # test_img[row,col] = 1
        
    # plt.imshow(noback_img[:,:,2])

    img_color = noback_img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_save = []

    pink = [] # predicted cell 중앙 좌표
    los = [] # size filter 에서 살아남는 contours 
    
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        if M['m00'] >= contour_thr and  M['m00'] <= 500: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            code = Point(cx,cy)
            if not(polygon is None):
                if code.within(polygon):
                    pink.append((cx,cy))
                    los.append(contours[i])
                    contours_save.append(contours[i])
            else:
                pink.append((cx,cy))
                los.append(contours[i])
                
    def dot_expand(height=None, width=None, dots=None):
        white2 = np.zeros((height,width,3))*np.nan
        white2[:,:] = [255,255,255]
        boxsize = 10
        for z in range(len(dots)):
            row = dots[z][1]
            col = dots[z][0]
            white2[np.max([row-boxsize, 0]) : np.min([row+boxsize, height]), \
                   np.max([col-boxsize, 0]) : np.min([col+boxsize, width])] = 0
        return white2

    dots = list(pink)
    white2 = dot_expand(height=height, width=width, dots=dots)
    predict_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        predict_area.append((w[1][i],w[0][i]))
    # plt.imshow(white2)

    co = []
    for i in range(len(positive_indexs)):
        row = positive_indexs[i][0]
        col = positive_indexs[i][1]
        code = Point(col, row)
        if not(polygon is None):
            if code.within(polygon):
                co.append((int(col),int(row)))
        else: co.append((int(col),int(row)))
        
    dots = list(co)
    white2 = dot_expand(height=height, width=width, dots=dots)
    cell_positive_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        cell_positive_area.append((w[1][i],w[0][i]))
        
    # if True:
            # TN
    dots = list(co)
    # ansbox_p = dot_expand(height=height, width=width, dots=dots)
    # ansbox = np.array(ansbox_p[:,:,0])
    # predbox = np.array(white2[:,:,0])
    # predbox2 = np.zeros(predbox.shape)
    # predbox2[np.where(predbox==0)] = 255
    

    Cell_n = len(co)
    # Predict_n = len(pink)
    
    FN = len(list(set(co) - set(predict_area)))
    TP = Cell_n - FN
    
    FP = len(list(set(pink) - set(cell_positive_area)))
    
    if np.min([FN, TP, FP]) < 0:
        print(threshold, contour_thr)
        import sys; sys.exit()
    
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2 * precision * recall / (precision + recall)
        # acc = (TP + TN) / (TP + TN + FP + FN)
    except: F1_score = 0; #  acc = 0
    
    # plt.imshow(predbox2)
    # plt.imshow(ansbox)
    
    msdict = {'los': los, 'co': co, 'tp': TP, 'fp': FP, 'fn': FN, 'pink': pink, 'contours_save': contours_save}
    
    return F1_score, msdict


#%% full ROI test

# img 받아서 crop하고 test후, 다시 재조립하고 한 imag 완성까지
path4 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save2\\generalization\\semilabel\\unetXY\\';
final_weightsave = path4 + 'total_total_final_unet.h5'
gc.collect(); tf.keras.backend.clear_session()
model_unet = model_unet_setup(output_bias=initial_bias)
model_unet.load_weights(final_weightsave)

i = 0
for i in range(len(tnlist)):
    
    msid = keylist[i]
    #
    t_im = dictionary[msid]['imread']
    marker_x = dictionary[msid]['marker_x']
    marker_y = dictionary[msid]['marker_y']
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    polygon = dictionary[msid]['polygon']
    points = dictionary[msid]['points']
    
    tim = np.array(t_im)
    rows = tim.shape[0]
    cols = tim.shape[1]
    bins = int(128/2)
    cs = 128
    
    row_for = range(0, rows-cs, bins)
    col_for = range(0, cols-cs, bins)

    Xunet_crop, Zunet_crop = [], []
    for row in row_for:
        for col in col_for:
            Xunet_crop.append(tim[row:row+cs, col:col+cs, :]/255)
            Zunet_crop.append([row, row+cs, col, col+cs])
            # print([row, row+cs, col, col+cs])
    Xunet_crop = np.array(Xunet_crop)
    
    yhat = model_unet.predict(Xunet_crop)
    
    allo = np.zeros((t_im.shape[0], t_im.shape[1])) * np.nan
    for j in range(len(yhat)):
        ix1, ix2, ix3, ix4 = Zunet_crop[j][0], Zunet_crop[j][1], Zunet_crop[j][2], Zunet_crop[j][3]
        tmp = np.nanmean([allo[ix1:ix2, ix3:ix4], yhat[j][:,:,1]], axis=0)
        allo[ix1:ix2, ix3:ix4] = tmp
        # plt.figure() 
        # plt.imshow(allo)
        # plt.imshow(yhat[210][:,:,1])
        
        # print(np.sum(np.isnan(allo)))
  
    # plt.imshow(tim)
    # plt.imshow(allo)
    
    threshold = 0
    
    for threshold in np.arange(-15, 0, 1):
        contour_thr = 80
        
        F1_score, msdict = get_F1_Unet(threshold=threshold, contour_thr=contour_thr,\
                   allo=allo, t_im=t_im, positive_indexs=positive_indexs, polygon=polygon)
        
        print(threshold, contour_thr, F1_score)
    

#%% generalization test

#%% dictionary_external load

add_dataset_loadpath = (['C:\\SynologyDrive\\study\\dy\\TH_안소라선생님\\resize_crop\\', \
                         'C:\\SynologyDrive\\study\dy\\56\\ground_truth\\'])

tnlist = []
for j in range(len(add_dataset_loadpath)):
    flist = os.listdir(add_dataset_loadpath[j])
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.jpg':
            tnlist.append(add_dataset_loadpath[j] + os.path.splitext(flist[i])[0])  
tnlist = list(set(tnlist))


dictionary_external = {}
for n_num in range(len(tnlist)):
    
    ms_basename = os.path.basename(tnlist[n_num])
    ms_dir = os.path.dirname(tnlist[n_num]) + '\\'
    msid = os.path.splitext(ms_basename)[0]

    df = pd.read_csv(ms_dir + ms_basename + '.csv')
    marker_x = np.array(df)[:,0]
    marker_y = np.array(df)[:,1]
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    
    im = img.imread(ms_dir + ms_basename + '.jpg')
    msdict = {'t_im': np.array(im), 'marker_x': marker_x, \
              'marker_y': marker_y, 'polygon': None, 'points': None, 'positive_indexs': positive_indexs}
    
    dictionary_external[msid] = msdict

#%%
keylist2 = list(dictionary_external.keys())
for i in range(len(keylist2)):
    msid = keylist2[i]
    
    t_im = dictionary_external[msid]['t_im']
    marker_x = dictionary_external[msid]['marker_x']
    marker_y = dictionary_external[msid]['marker_y']
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    polygon = dictionary_external[msid]['polygon']
    points = dictionary_external[msid]['points']
    
    tim = np.array(t_im)
    rows = tim.shape[0]
    cols = tim.shape[1]
    bins = int(128/2)
    cs = 128
    
    row_for = range(0, rows-cs, bins)
    col_for = range(0, cols-cs, bins)

    Xunet_crop, Zunet_crop = [], []
    for row in row_for:
        for col in col_for:
            Xunet_crop.append(tim[row:row+cs, col:col+cs, :])
            Zunet_crop.append([row, row+cs, col, col+cs])
            # print([row, row+cs, col, col+cs])
    Xunet_crop = np.array(Xunet_crop)
    
    yhat = model_unet.predict(Xunet_crop)
    
    allo = np.zeros((t_im.shape[0], t_im.shape[1])) * np.nan
    for j in range(len(yhat)):
        ix1, ix2, ix3, ix4 = Zunet_crop[j][0], Zunet_crop[j][1], Zunet_crop[j][2], Zunet_crop[j][3]
        tmp = np.nanmean([allo[ix1:ix2, ix3:ix4], yhat[j][:,:,1]], axis=0)
        allo[ix1:ix2, ix3:ix4] = tmp

    # plt.imshow(tim)
    # plt.imshow(allo)
    
    threshold = 0
    mssave = []
    for threshold in np.arange(-20, 0, 1):
        contour_thr = 10
        
        F1_score, msdict = get_F1_Unet(threshold=threshold, contour_thr=contour_thr,\
                   allo=allo, t_im=t_im, positive_indexs=positive_indexs, polygon=polygon)
        # print(threshold, contour_thr, F1_score)
        
        mssave.append(F1_score)
    print(msid, np.max(mssave))
    


# 내일 다시


                
#%%
epand_num = 6
def positive_expand(row=None, col=None, epand_num=epand_num, minimum_num=0):
    rowcol_list = []
    for rowi in range(-epand_num, epand_num+1):
        for coli in range(-epand_num, epand_num+1):
            distance = ((rowi**2 + coli**2))**0.5 
            if distance < epand_num + 0.1 and distance > minimum_num:
                rowcol_list.append([row+rowi, col+coli])
    return rowcol_list

d = epand_num + 0.1; gap=[]; d2 = int(d+1)
for row in range(-d2,d2):
    for col in range(-d2,d2):
        distance = np.sqrt(row**2 + col**2)
        if distance < d:
            gap.append((row,col))
            
d = 30; gap2=[]
for row in range(-d,d):
    for col in range(-d,d):
        distance = np.sqrt(row**2 + col**2)
        if distance <= d:
            gap2.append((row,col))


# mask = np.zeros((SIZE_HF*2,SIZE_HF*2))
# rowcol_list = positive_expand(row=SIZE_HF, col=SIZE_HF, epand_num=epand_num)
# for i in range(len(rowcol_list)):
#     mask[rowcol_list[i][0], rowcol_list[i][1]] = 1
    
# truly index도 추가하도록 하고
# truly index와 semi positive 합집합에서, 중심좌표 기준으로 (혹은 경계면 기준으로) crop에 들어오는 모든 pixel에 대하여 negtive로 select하도록 수정

# 중심좌표 부터 정리하고, exclustion 한다음 (기존 자료는 save)
# 살아남은 좌표들 주변 부 negativ따고, 기존자료랑 겹치면 negative도 제외 시킴

# im = t_im
# semi_msdict = msdict
# msdict = dictionary[msid]
#%%
def msXYZgen_semi(msdict=None, semi_msdict=None, rlist=None):
    rlist = rlist
    marker_x = msdict['marker_x'][rlist]
    marker_y = msdict['marker_y'][rlist]
    im = msdict['imread']
    polygon = msdict['polygon']
    
    SIZE_HF = 14
    chnum = im.shape[2]
    
    # prediction, ground truth merge
    contours_save = semi_msdict['contours_save']
    
    semi_positive_coordinates = []
    for i in range(len(contours_save)):
        msout = find_inner_coordinate(contours=contours_save[i])
        semi_positive_coordinates.append(msout)
        
    FN = []
    for i in range(len(marker_x)):
        cx = marker_x[i]
        cy = marker_y[i]
        
        code = Point(cx,cy)
        if code.within(polygon):
            sw = True
            for j in range(len(semi_positive_coordinates)):
                if (cx, cy) in semi_positive_coordinates[j]:
                    sw = False; break
            if sw: FN.append([cx, cy])
    
    print(len(marker_x), len(FN))
    # padding
    
    im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, chnum)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = im
    
    # FN add
    X, Y, Z = [], [], []
    positive_index = []
    
    for i in range(len(FN)):
        row, col = FN[i][1], FN[i][0]
        code = Point(col,row)
        if code.within(polygon):
            rowcol_list = positive_expand(row=row,col=col)
            for ei in range(len(rowcol_list)):
                row2 = rowcol_list[ei][0] + SIZE_HF
                col2 = rowcol_list[ei][1] + SIZE_HF
            
                crop = np.array(im_padding[row2-SIZE_HF:row2+SIZE_HF, col2-SIZE_HF:col2+SIZE_HF, :])
                if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
                    for ch in range(chnum):
                        crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                    X.append(np.array(crop))
                    Y.append([0,1])
                    Z.append(list([row2, col2]))
                positive_index.append((row2, col2))
        
    # prdiction add
    for i in range(len(semi_positive_coordinates)):
        for j in range(len(semi_positive_coordinates[i])):
            row = semi_positive_coordinates[i][j][1]
            col = semi_positive_coordinates[i][j][0]
            
            code = Point(col,row)
            if code.within(polygon):
                row2 = row + SIZE_HF
                col2 = col + SIZE_HF

                crop = np.array(im_padding[row2-SIZE_HF:row2+SIZE_HF, col2-SIZE_HF:col2+SIZE_HF, :])
                if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
                    for ch in range(chnum):
                        crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                    X.append(np.array(crop))
                    Y.append([0,1])
                    Z.append(list([row2, col2]))
                positive_index.append((row2, col2))

    # negative label
    
    positive_center_index = []
    for i in range(len(semi_msdict['pink'])):
        positive_center_index.append(list(semi_msdict['pink'][i]))
    for i in range(len(FN)):
        positive_center_index.append(list(FN[i]))
    
    for i in tqdm(range(len(positive_center_index))):
        col = positive_center_index[i][0]
        row = positive_center_index[i][1]
        
        rowcol_list = positive_expand(row=row, col=col, epand_num=12, minimum_num=7)
        for ei in range(len(rowcol_list)):
            row2 = rowcol_list[ei][0] + SIZE_HF
            col2 = rowcol_list[ei][1] + SIZE_HF
            
            if not((row2, col2) in positive_index):
                crop = np.array(im_padding[row2-SIZE_HF:row2+SIZE_HF, col2-SIZE_HF:col2+SIZE_HF, :])
                if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
                    for ch in range(chnum):
                        crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                    X.append(np.array(crop))
                    Y.append([1,0])
                    Z.append(list([row2, col2]))
            # positive_index.append((row2, col2))

    if False: # 확인용
        positive = np.where(np.array(Y)[:,1]==1)[0]
        negative = np.where(np.array(Y)[:,0]==1)[0]

        im2 = np.array(im)

        for y in positive:

            im2[Z[y][0] - SIZE_HF, Z[y][1] - SIZE_HF, 2] = 255 
            
        pil_image = Image.fromarray(im2)
        pil_image.save('C:\\Temp\\test_positive.jpg')

        for y in negative:
            im2[Z[y][0] - SIZE_HF, Z[y][1] - SIZE_HF, 1] = 255
            
        plt.figure(); plt.imshow(im)
        plt.figure(); plt.imshow(im2)
        
        pil_image = Image.fromarray(im2)
        pil_image.save('C:\\Temp\\test_both.jpg')
        
    print(np.mean(np.array(Y), axis=0))
        
    msdict = {'X':X, 'Y':Y}
    return msdict




#%%
path1 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post1\\'
path2 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\'
path3 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post1\\predict\\'


path1 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post2\\'
path2 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\' # 최초로 고정 (rlist legacy)
path3 = path1 + 'predict\\'


n_num = 0
for n_num in tqdm(range(len(keylist))):
    msid = keylist[n_num]
    psave = path1 + 'data_52_ms_XYZ_semi_' + str(msid) + '.pickle'
    psave2 = path2 + 'data_52_ms_XYZ_' + str(msid) + '.pickle'
    
    if not(os.path.isfile(psave)) or False:
        
        with open(psave2, 'rb') as file:
            msdict_XY1 = pickle.load(file)
            rlist = msdict_XY1['rlist']

        msdict = dictionary[msid]

        psavepath = path3 + msid + '_semilabel.pickle'
        with open(psavepath, 'rb') as file:
            msdict2 = pickle.load(file)
        
        msdict3 = msXYZgen_semi(msdict=msdict, semi_msdict=msdict2, rlist=rlist)
        with open(psave, 'wb') as file:
            pickle.dump(msdict3, file)
            print(psave, '저장되었습니다.')

#%% keras setup
# import tensorflow_addons as tfa
# metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

# def model_setup(xs=None, ys=None, lr=1e-4, lnn=7):
#     from tensorflow.keras.optimizers import Adam
#     model = models.Sequential()

#     model.add(layers.Conv2D(2**lnn, (4, 4), activation='relu', input_shape=xs))
#     model.add(layers.MaxPooling2D((2, 2)))
    
#     model.add(layers.Conv2D(2**lnn, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
    
#     model.add(layers.Conv2D(2**lnn, (2, 2), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))

#     lnn2 = lnn-1
#     lnn3 = lnn-2
    
#     model.add(layers.Flatten())
#     model.add(layers.Dense(2**lnn2, activation='relu' ))
#     model.add(layers.Dense(2**lnn2, activation='relu' ))
#     model.add(layers.Dense(2**lnn2, activation='relu' ))
#     model.add(layers.Dense(2**lnn3, activation='relu' ))
#     model.add(layers.Dense(2**lnn3, activation='relu' ))
#     model.add(layers.Dense(2**lnn3, activation='relu' ))
#     model.add(layers.Dense(2**lnn3, activation='sigmoid') )
#     model.add(layers.Dense(ys, activation='softmax'))

#     model.compile(optimizer=Adam(learning_rate=lr, decay=1e-3, beta_1=0.9, beta_2=0.999), \
#                   loss='categorical_crossentropy', metrics=metric) 
    
#     return model

# model = model_setup(xs=(29,29,3), ys=(2), lnn=7)
# print(model.summary())

#%%
mainpath = path1
cv = 0
lnn = 7
repeat = 1
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)

xyz_loadpath = mainpath
xyz_loadpath2 = 'F:'  + xyz_loadpath[2:]
cvlistsavepath = xyz_loadpath + 'cvlist2.pickle'

if not(os.path.isfile(cvlistsavepath)):
    xyz_loadpath = mainpath
    flist = os.listdir(xyz_loadpath)
    nlist = []
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.pickle' and flist[i][:7] == 'data_52':
            nlist.append(flist[i])
    nlist = list(set(nlist))
    
    rlist = list(range(len(nlist))); 
    random.seed(0); random.shuffle(rlist)    
    nlist = np.array(nlist)[rlist]
    cvnum = 20
    divnum = len(rlist) / cvnum
    cvlist = []
    for cv in range(cvnum):
        cvlist.append(list(range(int(round(cv*divnum)), int(round((cv+1)*divnum)))))
    print(cvlist)
    
    cvlist_msid = []
    for i in range(cvnum):
        cvlist_msid.append(nlist[cvlist[i]])
        
    msdict = {'tnlist': nlist, 'cvlist_msid': cvlist_msid}
    with open(cvlistsavepath, 'wb') as f:  # Python 3: open(..., 'rb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(cvlistsavepath, '저장되었습니다.')

with open(cvlistsavepath, 'rb') as file:
    msdict = pickle.load(file)
    cvlist_msid = msdict['cvlist_msid']
    tnlist = msdict['tnlist']
    
#%% model training

for cv in range(len(cvlist_msid)):
# for cv in range(0, len(cvlist)):
    # 1. weight
    print('cv', cv)
    weight_savename = 'cv_' + str(cv) + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename

    if not(os.path.isfile(final_weightsave)) or False:
        telist = cvlist_msid[cv]
        trlist = list(set(tnlist) - set(telist))
        
        model = model_setup(xs=(28, 28, 3), ys=2, lnn=lnn)
        if cv==0: print(model.summary())
        
        resetsw = True
        for i in range(len(trlist)):
            start = time.time()
            gc.collect(); tf.keras.backend.clear_session()
            psave = xyz_loadpath + trlist[i]
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                
            if resetsw:
                Xtr = np.array(msdict['X'])
                Ytr = np.array(msdict['Y'])
                resetsw = False
            else:
                Xtr = np.concatenate((Xtr, np.array(msdict['X'])), axis=0)
                Ytr = np.concatenate((Ytr, np.array(msdict['Y'])), axis=0)
            print('i', i, Xtr.shape)
            
            if (Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]) > 470400000 or i == len(trlist)-1:
                resetsw = True
                hist = model.fit(Xtr, Ytr, epochs=4, verbose=1, batch_size = 2**6)
       
        model.save_weights(final_weightsave)
        gc.collect()
        tf.keras.backend.clear_session()

#%% # 2. test data prep

forlist3 = [0]
for cv in range(len(cvlist_msid)): # range(len(cvlist)):
    # common load
    print('cv', cv)
    weight_savename = 'cv_' + str(cv) + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename
    
    telist = cvlist_msid[cv]

    n_num = telist[0]
    for n_num in telist:
        msid = n_num[20:-7]
        # test_image_no = 'bvPLA2 1-1.JPG_resize_crop'
        
        test_image_no = msid
        print(test_image_no)
        psave = weight_savepath + 'sample_n_' + test_image_no + '.pickle'
    
        # width = dictionary[test_image_no]['width']
        # height = dictionary[test_image_no]['length']
        
        t_im = dictionary[test_image_no]['imread']
        # t_im = np.mean(t_im, axis=2)
        marker_x = dictionary[test_image_no]['marker_x']
        marker_y = dictionary[test_image_no]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[test_image_no]['polygon']
        points = dictionary[test_image_no]['points']
    
        im = np.array(t_im)
        im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, 3)) * 255
        im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = im
        marker_y2 = marker_y + SIZE_HF
        marker_x2 = marker_x + SIZE_HF
        width = im_padding.shape[1]
        height = im_padding.shape[0]
    
        # 2. test data
        if not(os.path.isfile(psave)):
            import time; start = time.time()
            print('prep test data', cv)
            model = model_setup(xs=(28, 28, 3), ys=2, lnn=lnn)
            model.load_weights(final_weightsave)
            
            rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
            rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
            colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
            colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
            
            testarea = []
            for row in range(rowmin, rowmax):
                for col in range(colmin, colmax):
                    code = Point(col,row)
                    if code.within(polygon):
                        testarea.append([row, col])

            yhat_save = []
            z_save = []
            
            divnum = 10
            forlist = list(range(len(testarea)))
            div = int(len(forlist)/divnum)

            for div_i in range(divnum):
                print('div', div_i)
                if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
                elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
            
                X_total_te = []
                Z_total_te = []
                for i in forlist_div:
                    row, col = testarea[i][0], testarea[i][1]

                    crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                    for ch in range(3):
                        crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                        # plt.imshow(crop[:,:,ch])
                    X_total_te.append(np.array(crop))
                    Z_total_te.append([row-SIZE_HF, col-SIZE_HF])

                if len(X_total_te) > 0: # polygon ROI 때문에 필요함
                    X_total_te = np.array(X_total_te)
                    yhat = model.predict(X_total_te, verbose=1, batch_size = 2**6)
                    yhat_save += list(yhat[:,1])
                    z_save += Z_total_te
                        
            z_save = np.array(z_save)
            msdict = {'yhat_save': yhat_save, 'z_save': z_save}
            plt.figure(); plt.imshow(im_padding)
          
            print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
                
            if not(os.path.isfile(psave)) or False:
                with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
                    pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                    print(psave, '저장되었습니다.')


#%% 3. F1 score optimization

nlist_for = list(tnlist)

for n_num in range(len(nlist_for)):
    print('F1 score optimization', n_num)
    test_image_no = nlist_for[n_num][20:-7]
    psave = weight_savepath + 'sample_n_' + str(test_image_no) + '.pickle'
    psave2 = weight_savepath + 'F1_parameters_' + str(test_image_no) + '.pickle'
    if not(os.path.isfile(psave2)) and os.path.isfile(psave):
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            yhat_save = msdict['yhat_save']
            z_save = msdict['z_save']
            
        t_im = dictionary[test_image_no]['imread']
        t_im = np.mean(t_im, axis=2)
        marker_x = dictionary[test_image_no]['marker_x']
        marker_y = dictionary[test_image_no]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[test_image_no]['polygon']
        
        #%
        if False:
            threshold = 0.5; contour_thr = 100
            F1_score, _  = get_F1(threshold=0.5, contour_thr=40,\
                       yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
            print(F1_score)
        
        # optimize threshold, contour_thr
        import time; start = time.time()  # 시작 시간 저장
        
        # @ray.remote
        def ray_F1score_cal(forlist_cpu, yhat_save=yhat_save, \
                            positive_indexs=None, z_save=z_save, t_im=None, polygon=None):
            
            def F1_monte(threshold=None, s=None, e=None):
                F1_score_s, F1_score_e = None, None
                
                if not s is None:
                    F1_score_s, _ = get_F1(threshold=threshold, contour_thr=s,\
                                yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
                if not e is None:
                    F1_score_e, _ = get_F1(threshold=threshold, contour_thr=e,\
                                yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
                        
                return F1_score_s, F1_score_e
            
            mssave = []
            contour_thr_list = list(range(50,300,1))
            for threshold in forlist_cpu:
                
                s = contour_thr_list[0]
                e = contour_thr_list[-1]
                F1_score_s, F1_score_e = F1_monte(threshold=threshold, s=s, e=e)
                
                cnt = 0
                for tmp in range(100):
                    if s == e or cnt == 5: break
                    if F1_score_e == 0 and F1_score_s == 0: cnt += 1
                    # print(s, e)
                    # print(F1_score_s, F1_score_e)
                    
                    img_thr = int(np.mean([s, e]))
                    
                    pre_s = int(s); pre_e = int(e)
                    
                    if F1_score_s == F1_score_e:
                        s = int(np.mean([s, img_thr]))
                        e = int(np.mean([e, img_thr]))
                        F1_score_s, F1_score_e = F1_monte(threshold=threshold, s=s, e=e)
                    elif F1_score_s > F1_score_e:
                        e = int(np.mean([e, img_thr]))
                        _, F1_score_e = F1_monte(threshold=threshold, s=None, e=e)
                    elif F1_score_s < F1_score_e:
                        s = int(np.mean([s, img_thr]))
                        F1_score_s, _ = F1_monte(threshold=threshold, s=s, e=None)
                        
                    if pre_s == s and pre_e == e: cnt +=1
                        
                mssave.append([threshold, s, F1_score_s])
                print([threshold, s, F1_score_s])
            return mssave
        
        output_ids = []; 
        
        tresholds_list = np.round(np.arange(0.5,0.99,0.025), 3)
        forlist = list(tresholds_list)
        pre_F1 = -1
        for forlist_cpu in forlist:
            mssave = ray_F1score_cal([forlist_cpu], yhat_save=yhat_save, positive_indexs=positive_indexs, \
                                                     t_im=t_im, polygon=polygon)
            output_ids.append(mssave[0])
            if mssave[0][-1] < pre_F1: break
            pre_F1 = mssave[0][-1]
            
        tresholds_list = np.round(np.arange(0.49,0,-0.025), 3)
        forlist = list(tresholds_list)
        pre_F1 = -1
        for forlist_cpu in forlist:
            mssave = ray_F1score_cal([forlist_cpu], yhat_save=yhat_save, positive_indexs=positive_indexs, \
                                                     t_im=t_im, polygon=polygon)
            output_ids.append(mssave[0])
            if mssave[0][-1] < pre_F1: break
            pre_F1 = mssave[0][-1]

        mssave = np.array(output_ids)
        mix = np.argmax(mssave[:,2])
        result = [cv, test_image_no] + list(mssave[mix,:])
        print('\n', 'max F1 score', result)
        
        if not(os.path.isfile(psave2)) or True:
            with open(psave2, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                print(psave2, '저장되었습니다.')
        gc.collect()
        t1 = time.time() - start; print('\n', "time F1 optimize :", t1)  # 현재시각 - 시작시간 = 실행 시간

        if False: # 시각화
            threshold = mssave[mix,0]
            contour_thr = mssave[mix,1]
            F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                        yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
            
            plt.imshow(msdict['predict_img'])
            plt.imshow(msdict['truth_img'])

#%% generalization - trining path setup

# path1 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post1\\'
# path2 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\'
# path3 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post1\\predict\\'

path1 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\post2\\'
path2 = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\pre\\' # 최초로 고정 (rlist legacy)
path3 = path1 + 'predict\\'

mainpath = path1
cv = 0
lnn = 7
repeat = 1
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)

xyz_loadpath = mainpath
xyz_loadpath2 = 'F:'  + xyz_loadpath[2:]
cvlistsavepath = xyz_loadpath + 'cvlist2.pickle'

if not(os.path.isfile(cvlistsavepath)):
    xyz_loadpath = mainpath
    flist = os.listdir(xyz_loadpath)
    nlist = []
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.pickle' and flist[i][:7] == 'data_52':
            nlist.append(flist[i])
    nlist = list(set(nlist))
    
    rlist = list(range(len(nlist))); 
    random.seed(0); random.shuffle(rlist)    
    nlist = np.array(nlist)[rlist]
    cvnum = 20
    divnum = len(rlist) / cvnum
    cvlist = []
    for cv in range(cvnum):
        cvlist.append(list(range(int(round(cv*divnum)), int(round((cv+1)*divnum)))))
    print(cvlist)
    
    cvlist_msid = []
    for i in range(cvnum):
        cvlist_msid.append(nlist[cvlist[i]])
        
    msdict = {'tnlist': nlist, 'cvlist_msid': cvlist_msid}
    with open(cvlistsavepath, 'wb') as f:  # Python 3: open(..., 'rb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(cvlistsavepath, '저장되었습니다.')

with open(cvlistsavepath, 'rb') as file:
    msdict = pickle.load(file)
    cvlist_msid = msdict['cvlist_msid']
    tnlist = msdict['tnlist']

#%% generalization
mainpath = 'E:\\dy\\dy_THcelldetection\\semi_XY_save\\generalization\\'
weight_savepath = mainpath + 'weightsave_7\\'; msFunction.createFolder(weight_savepath)

weight_savename = 'total_total_final_post2.h5'

final_weightsave = weight_savepath + weight_savename
# lnn = 7
if not(os.path.isfile(final_weightsave)):
    
    model = model_setup(xs=(28, 28, CHNUM), ys=2, lnn=LNN)
    if cv==0: print(model.summary())
    
    resetsw = True
    for i in range(len(tnlist)):
        start = time.time()
        gc.collect(); tf.keras.backend.clear_session()
        psave = xyz_loadpath + tnlist[i]
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            
        if resetsw:
            Xtr = np.array(msdict['X'])
            Ytr = np.array(msdict['Y'])
            resetsw = False
        else:
            Xtr = np.concatenate((Xtr, np.array(msdict['X'])), axis=0)
            Ytr = np.concatenate((Ytr, np.array(msdict['Y'])), axis=0)
        print('i', i, Xtr.shape)
        
        if (Xtr.shape[0] * Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3]) > 470400000 or i == len(tnlist)-1:
            resetsw = True
            hist = model.fit(Xtr, Ytr, epochs=4, verbose=1, batch_size = 2**6)
   
    model.save_weights(final_weightsave)
    gc.collect()
    tf.keras.backend.clear_session()

#%% dictionary_external load

add_dataset_loadpath = (['C:\\SynologyDrive\\study\\dy\\TH_안소라선생님\\resize_crop\\', \
                         'C:\\SynologyDrive\\study\dy\\56\\ground_truth\\'])

tnlist = []
for j in range(len(add_dataset_loadpath)):
    flist = os.listdir(add_dataset_loadpath[j])
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.jpg':
            tnlist.append(add_dataset_loadpath[j] + os.path.splitext(flist[i])[0])  
tnlist = list(set(tnlist))


dictionary_external = {}
for n_num in range(len(tnlist)):
    
    ms_basename = os.path.basename(tnlist[n_num])
    ms_dir = os.path.dirname(tnlist[n_num]) + '\\'
    msid = os.path.splitext(ms_basename)[0]

    df = pd.read_csv(ms_dir + ms_basename + '.csv')
    marker_x = np.array(df)[:,0]
    marker_y = np.array(df)[:,1]
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    
    im = img.imread(ms_dir + ms_basename + '.jpg')
    msdict = {'t_im': np.array(im), 'marker_x': marker_x, \
              'marker_y': marker_y, 'polygon': None, 'points': None, 'positive_indexs': positive_indexs}
    
    dictionary_external[msid] = msdict

#%%
def mstest_gen(premodel=None, msdict=None):
    t_im = msdict['t_im']
    marker_x = msdict['marker_x']
    marker_y = msdict['marker_y']
    polygon = msdict['polygon']
    # points = msdict['points']
    
    marker_y2 = marker_y + SIZE_HF
    marker_x2 = marker_x + SIZE_HF
    
    im_padding = np.ones((t_im.shape[0]+SIZE_HF*2, t_im.shape[1]+SIZE_HF*2, 3)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = t_im

    # plt.imshow(im_padding)
    # plt.imshow(t_im)

    width = im_padding.shape[1]
    height = im_padding.shape[0]

    import time; start = time.time()
    
    rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
    rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
    colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
    colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
    
    yhat_save = []
    z_save = []
    
    divnum = 40
    forlist = list(range(rowmin, rowmax))
    div = int(len(forlist)/divnum)

    div_i = 0
    for div_i in range(divnum):
        gc.collect()
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
                        for ch in range(chnum):
                            crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])

                        X_total_te.append(np.array(crop))
                        Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
                else:
                    crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                    for ch in range(chnum):
                        crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                    # crop = np.reshape(crop, (crop.shape[0], crop.shape[1], 1))
                    
                    X_total_te.append(np.array(crop))
                    Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
            
        if len(X_total_te) > 0: # polygon ROI 때문에 필요함
            X_total_te = np.array(X_total_te)
            yhat = premodel.predict(X_total_te, verbose=1, batch_size = batch_size)
            yhat_save += list(yhat[:,1])
            z_save += Z_total_te
            
    z_save = np.array(z_save)
    msdict2 = {'yhat_save': yhat_save, 'z_save': z_save}
    # plt.figure(); plt.imshow(im_padding)
  
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    gc.collect()
    tf.keras.backend.clear_session()
    
    return msdict, msdict2    

#%%

def ray_F1score_cal(forlist_cpu, yhat_save=None, \
                    positive_indexs=None, z_save=None, t_im=None, polygon=None):
    
    def F1_monte(threshold=None, s=None, e=None):
        F1_score_s, F1_score_e = None, None
        
        if not s is None:
            F1_score_s, _ = get_F1(threshold=threshold, contour_thr=s,\
                        yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
        if not e is None:
            F1_score_e, _ = get_F1(threshold=threshold, contour_thr=e,\
                        yhat_save=yhat_save, positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
                
        return F1_score_s, F1_score_e
    
    mssave = []
    contour_thr_list = list(range(50,300,1))
    for threshold in forlist_cpu:
        
        s = contour_thr_list[0]
        e = contour_thr_list[-1]
        F1_score_s, F1_score_e = F1_monte(threshold=threshold, s=s, e=e)
        
        cnt = 0
        for tmp in range(100):
            if s == e or cnt == 5: break
            if F1_score_e == 0 and F1_score_s == 0: cnt += 1
            # print(s, e)
            # print(F1_score_s, F1_score_e)
            
            img_thr = int(np.mean([s, e]))
            
            pre_s = int(s); pre_e = int(e)
            
            if F1_score_s == F1_score_e:
                s = int(np.mean([s, img_thr]))
                e = int(np.mean([e, img_thr]))
                F1_score_s, F1_score_e = F1_monte(threshold=threshold, s=s, e=e)
            elif F1_score_s > F1_score_e:
                e = int(np.mean([e, img_thr]))
                _, F1_score_e = F1_monte(threshold=threshold, s=None, e=e)
            elif F1_score_s < F1_score_e:
                s = int(np.mean([s, img_thr]))
                F1_score_s, _ = F1_monte(threshold=threshold, s=s, e=None)
                
            if pre_s == s and pre_e == e: cnt +=1
                
        mssave.append([threshold, s, F1_score_s])
        print([threshold, s, F1_score_s])
    return mssave

#%% test data gen, test, F1 optimization, csv save

# final_weightsave = 'E:\\cell_detection_en_revision\\20230209_en_revision_original\\weightsave_7\\cv_0_total_final.h5'
mssave2 = []
keylist = list(dictionary_external.keys())
for i in range(len(keylist)):    
    msid = keylist[i]
    msdict = dictionary_external[msid]
    gc.collect(); tf.keras.backend.clear_session()
    
    psave = weight_savepath + 'sample_n_' + msid + '.pickle'
    if not(os.path.isfile(psave)):
        model = model_setup(xs=(28, 28, CHNUM), ys=2, lnn=LNN)
        final_weightsave = final_weightsave
        model.load_weights(final_weightsave)
        premodel = model
        msdict, msdict2 = mstest_gen(premodel=premodel, msdict=msdict)
        
        
        with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(msdict2, f, pickle.HIGHEST_PROTOCOL)
            print(psave, '저장되었습니다.')
        
    psave2 = weight_savepath + 'F1_parameters_' + str(msid) + '.pickle'
    if (os.path.isfile(psave) and not(os.path.isfile(psave2))) or False:
        with open(psave, 'rb') as file:
            msdict2 = pickle.load(file)
            yhat_save = msdict2['yhat_save']
            z_save = msdict2['z_save']
            
        t_im = msdict['t_im']
        polygon = msdict['polygon']
        positive_indexs = msdict['positive_indexs']
        # plt.imshow(t_im)
            
        # f1 score optimize
        output_ids = []; 
        
        tresholds_list = np.round(np.arange(0.5,0.99,0.025), 3)
        forlist = list(tresholds_list)
        pre_F1 = -1
        for forlist_cpu in forlist:
            mssave = ray_F1score_cal([forlist_cpu], yhat_save=yhat_save, z_save=z_save, positive_indexs=positive_indexs, \
                                                     t_im=t_im, polygon=polygon)
            output_ids.append(mssave[0])
            if mssave[0][-1] < pre_F1: break
            pre_F1 = mssave[0][-1]
            
        tresholds_list = np.round(np.arange(0.49,0,-0.025), 3)
        forlist = list(tresholds_list)
        pre_F1 = -1
        for forlist_cpu in forlist:
            mssave = ray_F1score_cal([forlist_cpu], yhat_save=yhat_save, z_save=z_save, positive_indexs=positive_indexs, \
                                                     t_im=t_im, polygon=polygon)
            output_ids.append(mssave[0])
            if mssave[0][-1] < pre_F1: break
            pre_F1 = mssave[0][-1]

        mssave = np.array(output_ids)
        mix = np.argmax(mssave[:,2])
        result = [0, msid] + list(mssave[mix,:])
        print('\n', 'max F1 score', result)
        
        if not(os.path.isfile(psave2)) or False:
            with open(psave2, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                print(psave2, '저장되었습니다.')
        gc.collect()
        
    if os.path.isfile(psave2):
        with open(psave, 'rb') as file:
            msdict2 = pickle.load(file)
            yhat_save = msdict2['yhat_save']
            z_save = msdict2['z_save']
            
        with open(psave2, 'rb') as file:
            result = pickle.load(file)
            
        threshold = result[2]
        contour_thr = result[3]
        t_im = msdict['t_im']
        polygon = msdict['polygon']
        positive_indexs = msdict['positive_indexs']
        
        F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                   yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
    
        predicted_cell_n = len(msdict['los'])
        tmp = [msid, len(msdict['co']), predicted_cell_n, F1_score, msdict['tp'], msdict['fp'], msdict['fn'], threshold, contour_thr]
        mssave2.append(tmp)
        print(F1_score)
    
print()
mssave2 = np.array(mssave2)
#%%
excel_save = weight_savepath + 'F1score_result_generalization_semi.xls'
print(excel_save)
mssave2_df = pd.DataFrame(np.array(mssave2[:,1:], dtype=float), \
             columns=['Human #', 'Model #', 'F1_score', 'TP', 'FP', 'FN', 'threshold', 'contour_thr'], \
             index=mssave2[:,0])
mssave2_df.to_excel(excel_save)
















