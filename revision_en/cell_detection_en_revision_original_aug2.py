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


import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction

#%% rawdata_dict gen

mainpath = 'C:\\SynologyDrive\\study\\dy\\image_data\\processed_imgs\\'
mainpath_common = 'C:\\SynologyDrive\\study\\dy\\image_data\\'
psave = mainpath + 'data_ms.pickle'

if not(os.path.isfile(psave)):
    # img load, id list gen
    img_list = []
    id_list = []
    
    path = str(mainpath)
    cnt = 0
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                print("%s/%s" % (path, filename))
                
                filepath = path + '\\' + filename
                msid = filename[:-4]
                id_list.append([msid, cnt]); cnt += 1
                img_list.append([msid, filepath])
    img_list = np.array(img_list)
    id_list = np.array(id_list)
    
    # marker load
    marker_list = []
    path = mainpath_common + 'markers\\'
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                print("%s/%s" % (path, filename))
                
                filepath = path + '\\' + filename
                msid = filename[12:-4]
                if not(msid in id_list[:,0]): print(filename, 'missing')
                marker_list.append([msid, filepath])
    marker_list = np.array(marker_list)
    
    # ROI load (will be added)
    
    ROI_list = []
    path = mainpath_common + 'roiBorder\\'
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                print("%s/%s" % (path, filename))
                
                filepath = path + '\\' + filename
                msid = filename[10:-11]
                if not(msid in id_list[:,0]): print(filename, 'missing')
                ROI_list.append([msid, filepath])
    ROI_list = np.array(ROI_list)
    
    # gen dict from key(=id)
    dic = {}
    for i in tqdm(range(len(id_list))):
        msid = id_list[i][0]
        
        # if msid == 's210331_3L':
        #     import sys;sys.exit()
        
        # image
        idix = np.where(img_list[:,0] == msid)[0][0]
        im = img.imread(img_list[idix, 1])
        image = Image.open(img_list[idix, 1])
        width = image.size[0]
        length = image.size[1]
        
        # ROI
        try:
            idix = np.where(ROI_list[:,0] == msid)[0][0]
            df = np.array(pd.read_csv(ROI_list[idix, 1]))
            points = []
            for j in range(len(df)):
                points.append([int(df[j,0])*2, int(df[j,1])*2])
            polygon = Polygon(points)
        except:
            print(msid)
        
        # marker
        idix = np.where(marker_list[:,0] == msid)[0][0]
        df = np.array(pd.read_csv(marker_list[idix, 1]))
        marker_x = np.array(df[:,2], dtype=int)
        marker_y = np.array(df[:,3], dtype=int)
    
        r_dict = {'marker_x':marker_x,
                  'marker_y':marker_y,
                  'imread': im,
                  'width': width,
                  'length': length,
                  'points': points,
                  'polygon': polygon}
        
        dic[msid] = r_dict
    
    with open(psave, 'wb') as file:
        pickle.dump(dic, file)
        print(psave, '저장되었습니다.')
        
with open(psave, 'rb') as file:
    dictionary = pickle.load(file)
keylist = list(dictionary.keys())

#%% """HYPERPARAMETERS"""

NCT = 50 # INTENSITY 조정값

SIZE_HF = 14 # crop size: 29 x 29 
SIZE = (SIZE_HF*2)+1

#%%

def im_mean(crop):
    import cv2
    import numpy as np
    
    piece = (crop).astype(np.uint8)
    piece =cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
    piece=cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    p = cv2.mean(piece)
    return p[0]

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
        ansbox3 = ((ansbox - predbox2) == 255)
        # plt.imshow(ansbox3)
        TN = np.sum(ansbox3)
    
        Cell_n = len(co)
        Predict_n = len(pink)
        TP = Cell_n - len(list(set(co) - set(predict_area)))
        FP = Predict_n - TP
        FN = Cell_n - TP
    
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2 * precision * recall / (precision + recall)
        # acc = (TP + TN) / (TP + TN + FP + FN)
    except: F1_score = 0; #  acc = 0
    
    # plt.imshow(predbox2)
    # plt.imshow(ansbox)
    
    msdict = {'los': los, 'co': co, 'tp': TP, 'fp': FP, 'fn': FN, \
              'predbox2': predbox2, 'ansbox': ansbox}
    
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


#%% ROI check
if False:
    for i in range(len(keylist)):
        test_image_no = keylist[i]
        t_im = dictionary[test_image_no]['imread']
        marker_x = dictionary[test_image_no]['marker_x']
        marker_y = dictionary[test_image_no]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[test_image_no]['polygon']
        points = dictionary[test_image_no]['points']
    
        pts = np.array([points],  np.int32)
        tmp = cv2.polylines(t_im, pts, True, (0,0,255),2)
        plt.figure()
        plt.imshow(tmp)
        plt.title(str(i) +'_'+ test_image_no)

#%% marker check
if False:
    for i in tqdm(range(len(id_list))):
        msid = id_list[i][0]
        idnum = int(id_list[i][1]) #  's210331_3L'
        
        width = dictionary[msid]['width']
        height = dictionary[msid]['length']
        t_im = dictionary[msid]['imread']
        marker_x = dictionary[msid]['marker_x']
        marker_y = dictionary[msid]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[msid]['polygon']
        points = dictionary[msid]['points']

        #%
        plt.imshow(t_im)
        
        pts = np.array([points],  np.int32)
        tmp = cv2.polylines(t_im, pts, True, (0,0,255),2)
        for nn in range(len(positive_indexs)):
            tmp = cv2.circle(tmp, [marker_x[nn], marker_y[nn]], 3, (0,255,255), -1)  
        plt.imshow(tmp)
        plt.savefig('C:\\SynologyDrive\\study\\dy\\52\\figsave2\\'+id_list[i][0]+ '-result.jpg', dpi = 300, 
                    bbox_inches = 'tight', 
                    pad_inches = 0)

#%%
epand_num = 6
def positive_expand(row=None, col=None, epand_num=epand_num):
    rowcol_list = []
    for rowi in range(-epand_num, epand_num+1):
        for coli in range(-epand_num, epand_num+1):
            if ((rowi**2 + coli**2))**0.5 < epand_num + 0.1:
                rowcol_list.append([row+rowi, col+coli])
    return rowcol_list

mask = np.zeros((SIZE_HF*2,SIZE_HF*2))
rowcol_list = positive_expand(row=SIZE_HF, col=SIZE_HF, epand_num=epand_num)
for i in range(len(rowcol_list)):
    mask[rowcol_list[i][0], rowcol_list[i][1]] = 1
# plt.imshow(mask)
    
# d = positive_expand(row=100, col=100)
# len(d)
from PIL import Image,ImageEnhance 
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
    #
    marker_y2 = marker_y + SIZE_HF
    marker_x2 = marker_x + SIZE_HF
    
    # marker 기준 rectangle ROI 생성
    # market 최 외곽 기준 +- 50 pix로 ROI 가져감

    # padding img gen & 좌표변경
    im_uint8 = Image.fromarray(np.uint8(im))
    
    bright_list = [0.9, 1.0, 1.1]
    contrast_list = [0.9, 1.0, 1.1]


    if False:
        for bright_var in [0.9, 1, 1.2, 2]:
            enh = ImageEnhance.Brightness(im_uint8).enhance(bright_var)
            print(bright_var, np.mean(np.array(enh)/220))
            plt.figure(); plt.imshow(np.array(enh))
            
        for con_var in [0.8, 0.9, 1.0, 1.1, 1.2]:
            enh = ImageEnhance.Contrast(im_uint8).enhance(con_var)
            print(con_var, np.mean(np.array(enh)), np.std(np.array(enh))/58.4)
            plt.figure(); plt.imshow(enh)
    
    timg_save = np.array(im)
    
    bright_var, con_var = 0.8, 0.9
    for bright_var in bright_list:
        en = ImageEnhance.Brightness(im_uint8).enhance(bright_var)
        for con_var in contrast_list:
            if bright_var == 1 and con_var == 1: continue
            enh = ImageEnhance.Brightness(en).enhance(con_var)       
            timg = np.array(enh)
            timg_save = np.concatenate((timg_save, timg), axis=2)
            
    print(timg_save.shape)
    chnum = timg_save.shape[2]
    positive_index = [] 
    
    im_padding = np.ones((im.shape[0]+SIZE_HF*2, im.shape[1]+SIZE_HF*2, chnum)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = timg_save
            
    for i in range(len(marker_x)):
        row = marker_y2[i]
        col = marker_x2[i]

        rowcol_list = positive_expand(row=row,col=col)
        for ei in range(len(rowcol_list)):
            row2 = rowcol_list[ei][0]
            col2 = rowcol_list[ei][1]
        
            crop = np.array(im_padding[row2-SIZE_HF:row2+SIZE_HF, col2-SIZE_HF:col2+SIZE_HF, :])
            if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
                for ch in range(chnum):
                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])

                X.append(np.array(crop))
                Y.append([0,1])
                Z.append(list([n, row, col]))
        positive_index.append((row, col))
       
    # negative label
    width = im_padding.shape[1]
    height = im_padding.shape[0]
    
    rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
    rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
    colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
    colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
    rectangle_roi_dict = {'rowmin': rowmin, 'rowmax': rowmax, 'colmin': colmin, 'colmax': colmax}
        
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
        
    total_index = gen_total_index(polygon=polygon, rectangle_roi_dict=rectangle_roi_dict)
         
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
        if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
            for ch in range(chnum):
               crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
               
            # crop = np.reshape(crop, (crop.shape[0], crop.shape[1], 1))
            X.append(np.array(crop))
            Y.append([1,0])
            Z.append(list([n, row, col]))

    rix = random.sample(negative_n2, int(len(negative_n2)/5))
    for j in rix:
        row = negative_index2[j][0]
        col = negative_index2[j][1]
        
        crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
        if crop.shape == (SIZE_HF*2, SIZE_HF*2, chnum) and not(np.isnan(np.mean(crop))):
            for ch in range(chnum):
               crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
               
            # crop = np.reshape(crop, (crop.shape[0], crop.shape[1], 1))
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

mainpath = 'D:\\cell_detection_en_revision\\20230209_en_revision_original_multi_input\\'
# mainpath = 'E:\\cell_detection_en_revision\\20230209_en_revision_original\\'
msFunction.createFolder(mainpath)
# mainpath = 'D:\\dy\\dy_THcelldetection\\XYZsave_20220913_original\\'

n_num = 0
for n_num in tqdm(range(len(keylist))):
    n = keylist[n_num]
    psave = mainpath + 'data_52_ms_XYZ_' + str(n) + '.pickle'
    
    if not(os.path.isfile(psave)) or False:
        marker_x = dictionary[n]['marker_x']
        marker_y = dictionary[n]['marker_y']
        im = dictionary[n]['imread']
        # im = np.mean(im, axis=2)
        msdict = msXYZgen(im=im, marker_x=marker_x, marker_y=marker_y)
        with open(psave, 'wb') as file:
            pickle.dump(msdict, file)
            print(psave, '저장되었습니다.')


#%% dataset1 only
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

#%%

cv = 0;
lnn = 7
repeat = 1
# weight_savepath = mainpath + 'weightsave_' + str(lnn) + \
#     '_' + str(repeat) + '\\'; msFunction.createFolder(weight_savepath)
    
# weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)

if False:
    df = pd.DataFrame(nlist)
    df = pd.concat((df, pd.DataFrame(cvlist)))
    df.to_csv('C:\\Temp\\20220919_dataset123_develop2\\nlist.csv')          

chnum = 27
#%% model training

for cv in range(len(cvlist)):
# for cv in range(0, len(cvlist)):
    # 1. weight
    print('cv', cv)
    weight_savename = 'cv_' + str(cv) + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename

    if not(os.path.isfile(final_weightsave)) or False:
        tlist = list(range(len(nlist)))
        telist = cvlist[cv]
        trlist = list(set(tlist) - set(telist))
        
        telist_name = []
        # X_te, Y_te, Z_te = [], [] ,[]
        for te in telist:
            gc.collect(); tf.keras.backend.clear_session()
            # psave = xyz_loadpath + nlist[te]
            telist_name.append(nlist[te])
        #     with open(psave, 'rb') as file:
        #         msdict = pickle.load(file)
        #         X_te += msdict['X']
        #         Y_te += msdict['Y']
        #         Z_te += msdict['Z']
        # X_te = np.array(X_te); Y_te = np.array(Y_te); Z_te = np.array(Z_te)
        # print(X_te.shape); print(Y_te.shape); print(np.mean(X_te[0]))
        
        
        psave_telist_name = weight_savepath + 'cv_' + str(cv) + '_cvlist.pickle'
        with open(psave_telist_name, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(telist_name, f, pickle.HIGHEST_PROTOCOL)
            print(psave_telist_name, '저장되었습니다.')
        
        # if False:
        #     for ms in [10, 1000,2000,5000]:
        #         plt.figure()
        #         plt.imshow(X_te[ms])
        
        model = model_setup(xs=(28, 28, chnum), ys=2, lnn=lnn)
        if cv==0: print(model.summary())
        
        epochs = 1; cnt=0
        for epoch in range(epochs):
            for n_num in trlist:
                import time; start = time.time()
                
                gc.collect(); tf.keras.backend.clear_session()
                psave = xyz_loadpath + nlist[n_num]
                with open(psave, 'rb') as file:
                    msdict = pickle.load(file)
                    X_tmp = np.array(msdict['X'])
                    Y_tmp = np.array(msdict['Y'])
                    Z_tmp = msdict['Z']
                    
                print(n_num, X_tmp.shape, np.mean(Y_tmp, axis=0), np.sum(Y_tmp, axis=0))
                
                # X_tmp = np.reshape(X_tmp, (len(X_tmp), 28,28,1))   # 임시
                hist = model.fit(X_tmp, Y_tmp, epochs=4, verbose=1, batch_size = 2**6)
                print("time :", time.time() - start)
                
                # cnt+=1
                # if cnt == 20 and False:
                #     cell = np.where(Y_tmp[:,1]==1)[0]
                #     yhat_cell = model.predict(X_tmp[cell], verbose=1, batch_size = 2**6)
                #     print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                #     noncell = np.where(Y_tmp[:,1]==0)[0]
                #     yhat_noncell = model.predict(X_tmp[noncell], verbose=0, batch_size = 2**6)
                #     print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                #     cnt = 0
                #     cell = np.where(Y_te[:,1]==1)[0]
                #     yhat_cell = model.predict(X_te[cell], verbose=1, batch_size = 2**6)
                #     print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                #     noncell = np.where(Y_te[:,1]==0)[0]
                #     yhat_noncell = model.predict(X_te[noncell], verbose=0, batch_size = 2**6)
                #     print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                    # print(msFunction.msROC(yhat_noncell[:,1], yhat_cell[:,1]))
                    
            # model.evaluate(X_te, Y_te, verbose=1, batch_size = 2**6)
        model.save_weights(final_weightsave)
        gc.collect()
        tf.keras.backend.clear_session()

#%% # 2. test data prep
        
# cv = 0;
# div_for_memory = 5 # 몇등분 할건지

if False: # tmp
    nlist = np.array(cvlist)[:,1]; sylist = []
    for i in range(len(nlist)):
        if nlist[i][-4:] == 'crop':
            sylist.append(i)
            
forlist3 = [0]
for cv in range(len(cvlist)): # range(len(cvlist)):
    # common load
    print('cv', cv)
    weight_savename = 'cv_' + str(cv) + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename
    
    telist = cvlist[cv]

    for n_num in telist:
        test_image_no = nlist[n_num][15:-7]
        # test_image_no = 'bvPLA2 1-1.JPG_resize_crop'
        
        print(test_image_no)
        psave = weight_savepath + 'sample_n_' + test_image_no + '.pickle'
        if not(os.path.isfile(psave)):
            gc.collect()
            tf.keras.backend.clear_session()
            # width = dictionary[test_image_no]['width']
            # height = dictionary[test_image_no]['length']
            
            t_im = dictionary[test_image_no]['imread']
            # t_im = np.mean(t_im, axis=2)
            marker_x = dictionary[test_image_no]['marker_x']
            marker_y = dictionary[test_image_no]['marker_y']
            positive_indexs = np.transpose(np.array([marker_y, marker_x]))
            polygon = dictionary[test_image_no]['polygon']
            points = dictionary[test_image_no]['points']
            
            marker_y2 = marker_y + SIZE_HF
            marker_x2 = marker_x + SIZE_HF
        
            
            # im = np.mean(im, axis=2) # 3d -> 2d
            #
            
            bright_list = [0.9, 1.0, 1.1]
            contrast_list = [0.9, 1.0, 1.1]
            
            im_uint8 = Image.fromarray(np.uint8(t_im))
            
            timg_save = np.array(t_im)
            bright_var, con_var = 0.8, 0.9
            for bright_var in bright_list:
                en = ImageEnhance.Brightness(im_uint8).enhance(bright_var)
                for con_var in contrast_list:
                    if bright_var == 1 and con_var == 1: continue
                    enh = ImageEnhance.Brightness(en).enhance(con_var)       
                    timg = np.array(enh)
                    timg_save = np.concatenate((timg_save, timg), axis=2)
            
            print(timg_save.shape)
            chnum = timg_save.shape[2]
            positive_index = [] 
            
            im_padding = np.ones((t_im.shape[0]+SIZE_HF*2, t_im.shape[1]+SIZE_HF*2, chnum)) * 255
            im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = timg_save

            width = im_padding.shape[1]
            height = im_padding.shape[0]
        
        # 2. test data
        
            import time; start = time.time()
            print('prep test data', cv)
            model = model_setup(xs=(28, 28, chnum), ys=2, lnn=lnn)
            model.load_weights(final_weightsave)
            
            rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
            rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
            colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
            colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
            
            yhat_save = []
            z_save = []
            
            divnum = 10
            forlist = list(range(rowmin, rowmax))
            div = int(len(forlist)/divnum)

            for div_i in range(divnum):
                print('div', div_i)
                if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
                elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
            
                X_total_te = []
                Z_total_te = []
                for row in forlist_div:
                    for col in range(colmin, colmax):
                        if not(polygon is None):
                            code = Point(col,row)
                            if code.within(polygon):
                                crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                                for ch in range(chnum):
                                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                                # crop = np.reshape(crop, (crop.shape[0], crop.shape[1], 1))
                                
                                  
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
                    yhat = model.predict(X_total_te, verbose=1, batch_size = 2**6)
                    yhat_save += list(yhat[:,1])
                    z_save += Z_total_te
                    
            z_save = np.array(z_save)
            msdict = {'yhat_save': yhat_save, 'z_save': z_save}
            # plt.figure(); plt.imshow(im_padding)
          
            print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
                
            if not(os.path.isfile(psave)) or False:
                with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
                    pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                    print(psave, '저장되었습니다.')


#%% 3. F1 score optimization

nlist_for = list(nlist)

for n_num in range(len(nlist_for)):
    print('F1 score optimization', n_num)
    test_image_no = nlist_for[n_num][15:-7]
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

# 20220910 여기까지 수정
#%%

# csv_savepath = 'C:\\Temp\\20220919_dataset123_develop2\\merge_save.csv'
# df = pd.read_csv(csv_savepath)
# merge_save = np.array(df)[:,-1]
# print('len(merge_save)', len(merge_save))
#%% to excel
# merge_save = []

nlist_for = list(nlist)
mssave2 = []
for n_num in range(len(nlist_for)):
    msid = nlist_for[n_num][15:-7]
    # idnum = int(id_list[i][1]) #  's210331_3L'
    
    # yhat_save, z_save
    psave = weight_savepath + 'sample_n_' + msid + '.pickle'
    psave2 = weight_savepath + 'F1_parameters_' + msid + '.pickle'
    
    c1 = os.path.isfile(psave)
    c2 = os.path.isfile(psave2)
    
    if c1 and c2:
        print(msid)
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            yhat_save = msdict['yhat_save']
            z_save = msdict['z_save']
        
        # threshold, contour_thr
        
        with open(psave2, 'rb') as file:
            result = pickle.load(file)
            
        threshold = result[2]
        contour_thr = result[3]
        # width = dictionary[msid]['width']
        # height = dictionary[msid]['length']
        t_im = dictionary[msid]['imread']
        marker_x = dictionary[msid]['marker_x']
        marker_y = dictionary[msid]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[msid]['polygon']
        points = dictionary[msid]['points']
        
        #%
        if False:
            t_im2 = Image.fromarray((t_im).astype(np.uint8))
            pts = np.array([points],  np.int32)
            tmp = cv2.polylines(t_im, pts, True, (0,0,255),2)
            plt.imshow(t_im)
            plt.imshow(tmp)
        #%
        
        F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                   yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
            
        predicted_cell_n = len(msdict['los'])
        
        tmp = [msid, len(msdict['co']), predicted_cell_n, F1_score, msdict['tp'], msdict['fp'], msdict['fn'], threshold, contour_thr]
        # merge_save.append(msid)
        mssave2.append(tmp)
        print()
        print(tmp)
        
        if False:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title(msid + ' _truth_img')
            plt.imshow(msdict['truth_img'])
            plt.subplot(2, 1, 2)
            plt.title('predcit_img')
            plt.imshow(msdict['predcit_img'])
            plt.tight_layout()
            figsave_path = 'C:\\SynologyDrive\\study\\dy\\52\\figsave\\' + msid + '.png'
            plt.savefig(figsave_path, dpi=200)
            plt.close()
    
mssave2 = np.array(mssave2)
print()
print('F1 score mean', np.mean(np.array(mssave2[:,3], dtype=float)))

a = np.array(mssave2[:,3], dtype=float)
vix = np.where(a!=0)[0]
np.mean(a[vix])

excel_save = weight_savepath + 'F1score_result.xls'
mssave2_df = pd.DataFrame(mssave2, columns=['ID', 'Human #', 'Model #', 'F1_score', 'TP', 'FP', 'FN', 'threshold', 'contour_thr'])
mssave2_df.to_excel(excel_save)



#%% to excel -> thr 0.5 고정
# merge_save = []

nlist_for = list(nlist)
mssave2 = []
for n_num in range(len(nlist_for)):
    msid = nlist_for[n_num][15:-7]
    # idnum = int(id_list[i][1]) #  's210331_3L'
    
    # yhat_save, z_save
    psave = weight_savepath + 'sample_n_' + msid + '.pickle'
    psave2 = weight_savepath + 'F1_parameters_' + msid + '.pickle'
    
    c1 = os.path.isfile(psave)
    c2 = os.path.isfile(psave2)
    
    if c1 and c2:
        print(msid)
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            yhat_save = msdict['yhat_save']
            z_save = msdict['z_save']
        
        # threshold, contour_thr
        
        with open(psave2, 'rb') as file:
            result = pickle.load(file)
            
        threshold = result[2]
        contour_thr = result[3]
        # width = dictionary[msid]['width']
        # height = dictionary[msid]['length']
        t_im = dictionary[msid]['imread']
        marker_x = dictionary[msid]['marker_x']
        marker_y = dictionary[msid]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[msid]['polygon']
        points = dictionary[msid]['points']
        
        #%
        if False:
            t_im2 = Image.fromarray((t_im).astype(np.uint8))
            pts = np.array([points],  np.int32)
            tmp = cv2.polylines(t_im, pts, True, (0,0,255),2)
            plt.imshow(t_im)
            plt.imshow(tmp)
        #%
        threshold = 0.5
        F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                   yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
            
        predicted_cell_n = len(msdict['los'])
        
        tmp = [msid, len(msdict['co']), predicted_cell_n, F1_score, msdict['tp'], msdict['fp'], msdict['fn'], threshold, contour_thr]
        # merge_save.append(msid)
        mssave2.append(tmp)
        print()
        print(tmp)
        
        if False:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title(msid + ' _truth_img')
            plt.imshow(msdict['truth_img'])
            plt.subplot(2, 1, 2)
            plt.title('predcit_img')
            plt.imshow(msdict['predcit_img'])
            plt.tight_layout()
            figsave_path = 'C:\\SynologyDrive\\study\\dy\\52\\figsave\\' + msid + '.png'
            plt.savefig(figsave_path, dpi=200)
            plt.close()
    
mssave2 = np.array(mssave2)
print()
print('F1 score mean', np.mean(np.array(mssave2[:,3], dtype=float)))

a = np.array(mssave2[:,3], dtype=float)
vix = np.where(a!=0)[0]
np.mean(a[vix])

excel_save = weight_savepath + 'F1score_result.xls'
mssave2_df = pd.DataFrame(mssave2, columns=['ID', 'Human #', 'Model #', 'F1_score', 'TP', 'FP', 'FN', 'threshold', 'contour_thr'])
mssave2_df.to_excel(excel_save)


#%%

def mstest(t_im=None, premodel=None, msdict=None):
    gc.collect()
    tf.keras.backend.clear_session()
    
    t_im = msdict['t_im']
    marker_x = msdict['marker_x']
    marker_y = msdict['marker_y']
    polygon = msdict['polygon']
    # points = msdict['points']
    
    marker_y2 = marker_y + SIZE_HF
    marker_x2 = marker_x + SIZE_HF
    
        
    bright_list = [0.9, 1.0, 1.1]
    contrast_list = [0.9, 1.0, 1.1]
    
    im_uint8 = Image.fromarray(np.uint8(t_im))
    
    timg_save = np.array(t_im)
    bright_var, con_var = 0.8, 0.9
    for bright_var in bright_list:
        en = ImageEnhance.Brightness(im_uint8).enhance(bright_var)
        for con_var in contrast_list:
            if bright_var == 1 and con_var == 1: continue
            enh = ImageEnhance.Brightness(en).enhance(con_var)       
            timg = np.array(enh)
            timg_save = np.concatenate((timg_save, timg), axis=2)
            
            print(timg_save.shape)
            chnum = timg_save.shape[2]
            # positive_index = [] 
            
    im_padding = np.ones((t_im.shape[0]+SIZE_HF*2, t_im.shape[1]+SIZE_HF*2, chnum)) * 255
    im_padding[SIZE_HF:-SIZE_HF, SIZE_HF:-SIZE_HF, :] = timg_save

    width = im_padding.shape[1]
    height = im_padding.shape[0]

    import time; start = time.time()
    
    rowmin = np.max([np.min(marker_y2) - 50, SIZE_HF])
    rowmax = np.min([np.max(marker_y2) + 50, height-SIZE_HF])
    colmin = np.max([np.min(marker_x2) - 50, SIZE_HF])
    colmax = np.min([np.max(marker_x2) + 50, width-SIZE_HF])
    
    yhat_save = []
    z_save = []
    
    divnum = 20
    forlist = list(range(rowmin, rowmax))
    div = int(len(forlist)/divnum)

    div_i = 0
    for div_i in range(divnum):
        print('div', div_i)
        if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
        elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
    
        X_total_te = []
        Z_total_te = []
        for row in forlist_div:
            for col in range(colmin, colmax):
                if not(polygon is None):
                    code = Point(col,row)
                    if code.within(polygon):
                        crop = np.array(im_padding[row-SIZE_HF:row+SIZE_HF, col-SIZE_HF:col+SIZE_HF, :])
                        for ch in range(chnum):
                            crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                        # crop = np.reshape(crop, (crop.shape[0], crop.shape[1], 1))
                        
                          
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
            yhat = premodel.predict(X_total_te, verbose=1, batch_size = 2**6)
            yhat_save += list(yhat[:,1])
            z_save += Z_total_te
            
    z_save = np.array(z_save)
    msdict2 = {'yhat_save': yhat_save, 'z_save': z_save}
    # plt.figure(); plt.imshow(im_padding)
  
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    
    return msdict, msdict2

#%% # 2. test data prep

## img load, ## resize + preproccessing
weight_savepath = 'D:\\cell_detection_en_revision\\20230209_en_revision_original_multi_input\\dataset3\\'
msFunction.createFolder(weight_savepath)

sy_loadpath = 'C:\\SynologyDrive\\study\dy\\56\\ground_truth\\'
flist = os.listdir(sy_loadpath)
nlist = []
for i in range(len(flist)):
    nlist.append(os.path.splitext(flist[i])[0])
nlist = list(set(nlist))

n_num = 0; mssave2 = []
for n_num in range(len(nlist)):
    msid = nlist[n_num]  
    df = pd.read_csv(sy_loadpath + msid + '.csv')
    marker_x = np.array(df)[:,0]
    marker_y = np.array(df)[:,1]
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    
    im = img.imread(sy_loadpath + msid + '.jpg')
    msdict = {'t_im': np.array(im), 'marker_x': marker_x, \
              'marker_y': marker_y, 'polygon': None, 'points': None, 'positive_indexs': positive_indexs}
    
    psave = weight_savepath + 'sample_n_' + msid + '.pickle'

    
    if not(os.path.isfile(psave)):
        if False:
            print('prep test data', cv)
            model = model_setup(xs=(28, 28, chnum), ys=2, lnn=lnn)
            final_weightsave = 'D:\\cell_detection_en_revision\\20230209_en_revision_original_multi_input\\weightsave_7\\cv_3_total_final.h5'
            model.load_weights(final_weightsave)
            premodel = model
            msdict, msdict2 = mstest(t_im=t_im, premodel=premodel, msdict=msdict)
            
            with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(msdict2, f, pickle.HIGHEST_PROTOCOL)
                print(psave, '저장되었습니다.')
        
    if os.path.isfile(psave):
        with open(psave, 'rb') as file:
            msdict2 = pickle.load(file)
            yhat_save = msdict2['yhat_save']
            z_save = msdict2['z_save']
            
        threshold = 0.5
        contour_thr = 100
        
        t_im = msdict['t_im']
        polygon = msdict['polygon']
        
        F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                   yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
    
        mssave2.append(F1_score)
        print(F1_score)
    
#  psave = weight_savepath + 'sample_n_' + test_image_no + '.pickle'

# msdict -> t_im, marker_x, marker_y, polygon, points
#%%



        
    if not(os.path.isfile(psave)) or False:
        with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
            print(psave, '저장되었습니다.')
            
            
    ##
    
























