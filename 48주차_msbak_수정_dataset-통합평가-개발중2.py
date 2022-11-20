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
mainpath = 'C:\\SynologyDrive\\study\\dy\\52\\'
psave = mainpath +'data_52_ms.pickle'

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
                
                filepath = path + '//' + filename
                msid = filename[:-4]
                id_list.append([msid, cnt]); cnt += 1
                img_list.append([msid, filepath])
    img_list = np.array(img_list)
    id_list = np.array(id_list)
    
    # marker load
    marker_list = []
    path = mainpath + 'markers\\'
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                print("%s/%s" % (path, filename))
                
                filepath = path + '//' + filename
                msid = filename[12:-4]
                if not(msid in id_list[:,0]): print(filename, 'missing')
                marker_list.append([msid, filepath])
    marker_list = np.array(marker_list)
    
    # ROI load (will be added)
    
    ROI_list = []
    path = mainpath + 'roiBorder\\'
    for (path, dir, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                print("%s/%s" % (path, filename))
                
                filepath = path + '//' + filename
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
        
with open(mainpath +'data_52_ms.pickle', 'rb') as file:
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
    
    vix = np.where(np.array(yhat_save) > threshold)[0]
    for i in vix:
        row = z_save[i][0]
        col = z_save[i][1]
        noback_img[row,col] = [0,0,255]
        
    # plt.imshow(noback_img)

    img_color = noback_img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pink = [] # predicted cell 중앙 좌표
    los = [] # size filter 에서 살아남는 contours 
    
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] >= contour_thr and  M['m00'] <= 500: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            code = Point(cx,cy)
            if not(polygon is None):
                if code.within(polygon):
                    pink.append((cx,cy))
                    los.append(cnt)
            else:
                pink.append((cx,cy))
                los.append(cnt)

    # predicted area 
    white = np.zeros((height,width,3))*np.nan
    white[:,:] = [255,255,255]
    for lo in los: # los : size filter 에서 살아남는 contours 
        cv2.drawContours(white, [lo], 0, (0, 0, 0), -2)  # black
    
    # plt.imshow(white)
    
    colored = [] # colored : predicted area 전체 표시 (작은 집단은 제외되어 있음)
    w = np.where(white[:,:,0]==0)
    for i in range(len(w[0])):
        colored.append((w[1][i],w[0][i]))
    
    # answer (positive) area
    co = []
    for i in range(len(positive_indexs)):
        row = positive_indexs[i][0]
        col = positive_indexs[i][1]

        code = Point(col, row)
        if not(polygon is None):
            if code.within(polygon):
                co.append((int(col),int(row)))
        else: co.append((int(col),int(row)))
  
    original_img = np.array(t_im, dtype=np.uint8)
    for nn in range(len(co)):
        cv2.circle(original_img, co[nn], 3, (0,255,255), -1)
        
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
                
    dots = co
    white2 = dot_expand(height=height, width=width, dots=dots)
    # plt.imshow(white2)    # ground truth area
    tparea = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        tparea.append((w[1][i],w[0][i]))     

    # predicted cells area
    # dots = pink

    fp_index = list(set(pink) - set(tparea))
    fp = len(fp_index)
    tp = len(pink) - fp
    fn = len(co) - tp

    # fn_list = list(set(co)-set(colored))
    # fn = len(fn_list)

    # tp_list = list(set(co)-set(fn_list))
    # tp = len(tp_list)
    # fp = len(los) - tp
    
    ###
    
    def ms_match_simulation(co, pink):
        co2 = np.array(co)
        pink2 = np.array(pink)
        
        dis_thr = 10
        dis_save = np.ones((len(pink2), len(co2))) * np.inf
        match_ix = np.zeros(len(pink2)) * np.nan
        for p in range(len(pink2)):
            for c in range(len(co2)):
                if not c in match_ix:
                    dis = np.sqrt(np.sum(np.square(pink2[p] - co2[c])))
                    dis_save[p,c] = dis
            
            close_enough = np.sum(dis_save[p,:] < dis_thr)
            if close_enough > 0:
                match_ix[p] = np.argmin(dis_save[p,:])
                # print(np.argmin(dis_save[p,:]), 'for', p)
                
        return match_ix
    
    co2 = np.array(co)
    pink2 = np.array(pink)
    mssave = []
    for epoch in range(20):
        random.seed(epoch)
        rd = random.randrange(2)
        
        np.random.seed(epoch)
        np.random.shuffle(co2)
        np.random.seed(epoch)
        np.random.shuffle(pink2)
        
        if rd == 0: co3 = np.array(co2); pink3 = np.array(pink2)
        if rd == 1: co3 = np.array(pink2); pink3 = np.array(co2)
        match_ix = ms_match_simulation(co3, pink3)
        TP = np.sum(np.isnan(match_ix)==0)
        FP = len(pink) - TP
        FN = len(co) - TP
        mssave.append([TP,FP,FN])
        
    TP = np.mean(np.array(mssave), axis=0)[0]
    FP = np.mean(np.array(mssave), axis=0)[1]
    FN = np.mean(np.array(mssave), axis=0)[2]
        
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2 * precision * recall / (precision + recall)
    except: F1_score = 0
    
    msdict = {'los': los, 'co': co, 'truth_img': white2, 'predict_img': white, 'tp': tp, 'fp': fp, 'fn': fn}
    
    return F1_score, msdict

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

    # predicted area 
    white = np.zeros((height,width,3))*np.nan
    white[:,:] = [255,255,255]
    for lo in los: # los : size filter 에서 살아남는 contours 
        cv2.drawContours(white, [lo], 0, (0, 0, 0), -2)  # black
    
    # plt.imshow(white)
    
    colored = [] # colored : predicted area 전체 표시 (작은 집단은 제외되어 있음)
    w = np.where(white[:,:,0]==0)
    for i in range(len(w[0])):
        colored.append((w[1][i],w[0][i]))
    
    # answer (positive) area
    co = []
    for i in range(len(positive_indexs)):
        row = positive_indexs[i][0]
        col = positive_indexs[i][1]

        code = Point(col, row)
        if not(polygon is None):
            if code.within(polygon):
                co.append((int(col),int(row)))
        else: co.append((int(col),int(row)))
  
    original_img = np.array(t_im, dtype=np.uint8)
    for nn in range(len(co)):
        cv2.circle(original_img, co[nn], 3, (0,255,255), -1)
        
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
                
    dots = co
    white2 = dot_expand(height=height, width=width, dots=dots)
    # plt.imshow(white2)    # ground truth area
    tparea = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        tparea.append((w[1][i],w[0][i]))     

    # predicted cells area
    dots = pink
    
    # white4 = dot_expand(height=height, width=width, dots=dots)
    # plt.imshow(white4)    # prediction area

    # co = ground truth center point
    # tparea = ground truth area
    # pink = prediction area center point
    # colored = prediction area total index

    pink = list(set(pink) - set(tparea))
    fn_list = list(set(co)-set(colored))
    fn = len(fn_list)
    
    # for nn in range(len(pink)):
    #     cv2.circle(original_img, pink[nn], 3, (255,0,255), -1) #################33
    # for nn in range(len(fn_list)):
    #     cv2.circle(original_img, fn_list[nn], 3, (0,255,20), -1)
    # for aa in range(len(points)):
    #     cv2.circle(original_img, points[aa],3,(0,0,0),-1) 
    # pts = np.array([points],  np.int32)
    # cv2.polylines(original_img, pts,True, (0,0,0), 1)
    
    ### detection save
    # de_roi = cv2.polylines(img_color,pts,True,(0,0,0),1)
    
    tp_list = list(set(co)-set(fn_list))
    tp = len(tp_list)
    fp = len(los) - tp
    
    # precision = tp/(tp+fp)
    # recall = tp / (tp+fn)
    # # score = 2*precision*recall / (precision+recall)
    # TP = tp
    # FP = fp
    # FN = fn
    
    try:
        precision = tp/(tp+fp)
        recall = tp / (tp+fn)
        F1_score = 2 * precision * recall / (precision + recall)
    except: F1_score = 0
    
    msdict = {'los': los, 'co': co, 'truth_img': white2, 'predict_img': white, 'tp': tp, 'fp': fp, 'fn': fn}
    
    return F1_score, msdict

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
    
# d = positive_expand(row=100, col=100)
# len(d)

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

    if False:
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

mainpath = 'C:\\Temp\\20220919_dataset123_develop2\\'
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

#%% XYZ gen - 박선영선생님 dataset

mainpath_dataset2 = 'C:\\Temp\\20220919_dataset123_develop2\\dataset2_sy\\'
msFunction.createFolder(mainpath_dataset2)

if True:
    if False: # resize
        imsavepath = 'C:\\SynologyDrive\\study\dy\\56\\'
        ex_mainpath = 'C:\\SynologyDrive\\study\\dy\\TH_박선영선생님\\TH\\'
        flist = os.listdir(ex_mainpath)
        for i in range(len(flist)):
            filename = flist[i]
            im = img.imread(ex_mainpath + filename)
            
            width = im.shape[1]
            height = im.shape[0]
            dst1 = cv2.resize(im, (int(round(width/2.5)), int(round(height/2.5))), \
                              interpolation=cv2.INTER_AREA)
    
            pil_image = Image.fromarray(dst1)
            pil_image.save(imsavepath + flist[i] + '_resize.jpg')
    
    # id 분리
    sy_loadpath = 'C:\\SynologyDrive\\study\\dy\\TH_박선영선생님\\crop_img_and_csv\\'
    flist = os.listdir(sy_loadpath)
    nlist = []
    for i in range(len(flist)):
        fname = os.path.splitext(flist[i])[0]
        if fname[-4:] == 'crop':
            nlist.append(fname)
            
    print('len(flist)', len(flist), '>>', 'len(nlist)', len(nlist), '<< 많이 줄어들면 error')
    nlist = list(set(nlist))
    
    # XYZ gen
    for n_num in tqdm(range(len(nlist))):
        n = nlist[n_num]
        psave = mainpath_dataset2 +'data_52_ms_XYZ_' + str(n) + '.pickle'
        
        if not(os.path.isfile(psave)) or False:
            
            df = pd.read_csv(sy_loadpath + n + '.csv')
            
            marker_x = np.array(df)[:,0]
            marker_y = np.array(df)[:,1]
            
            try: im = img.imread(sy_loadpath + n + '.jpg')
            except: im = img.imread(sy_loadpath + n + '.png')
            
            # im = np.mean(im, axis=2)
            msdict = msXYZgen(im=im, marker_x=marker_x, marker_y=marker_y)
            
            # np.array(msdict['X']).shape
            
            with open(psave, 'wb') as file:
                pickle.dump(msdict, file)
                print(psave, '저장되었습니다.')
     
    # dictionary에 추가
    for n_num in tqdm(range(len(nlist))):
        n = nlist[n_num]  
        df = pd.read_csv(sy_loadpath + n + '.csv')
        marker_x = np.array(df)[:,0]
        marker_y = np.array(df)[:,1]

        try: im = img.imread(sy_loadpath + n + '.jpg')
        except: im = img.imread(sy_loadpath + n + '.png')
    
        msdict = {'imread':im, 'marker_x':marker_x, 'marker_y':marker_y, 'polygon':None, 'points':None}
        dictionary[n] = msdict

#%% XYZ gen - 안소라선생님 dataset

mainpath_dataset3 = 'C:\\Temp\\20220919_dataset123_develop2\\dataset2_SORA\\'
msFunction.createFolder(mainpath_dataset3)

if True:
    if False: # resize
        imsavepath = 'C:\\SynologyDrive\\study\\dy\\TH_안소라선생님\\resize\\'; msFunction.createFolder(imsavepath)
        ex_mainpath = 'C:\\SynologyDrive\\study\\dy\\TH_안소라선생님\\'
        flist = os.listdir(ex_mainpath)
        for i in range(len(flist)):
            filename = flist[i]
            if os.path.splitext(filename)[1] == '.tif':
                im = img.imread(ex_mainpath + filename)
                
                width = im.shape[1]
                height = im.shape[0]
                
                
                # dpi_compensation = (2592/100) / (1600/40)
                
                comp = 1.048552668

                dst1 = cv2.resize(im, (int(round(width/comp)), int(round(height/comp))), \
                                  interpolation=cv2.INTER_AREA)
        
                pil_image = Image.fromarray(dst1)
                pil_image.save(imsavepath + flist[i] + '_resize.jpg')
    
    # id 분리
    sy_loadpath = 'C:\\SynologyDrive\\study\\dy\\TH_안소라선생님\\resize_crop\\'
    flist = os.listdir(sy_loadpath)
    nlist = []
    for i in range(len(flist)):
        nlist.append(os.path.splitext(flist[i])[0])
    nlist = list(set(nlist)); nlist.sort()
    
    # XYZ gen
    for n_num in tqdm(range(len(nlist))):
        n = nlist[n_num]
        psave = mainpath_dataset3 +'data_52_ms_XYZ_' + str(n) + '.pickle'
        
        if not(os.path.isfile(psave)) or False:
            
            df = pd.read_csv(sy_loadpath + n + '.csv')
            
            marker_x = np.array(df)[:,0]
            marker_y = np.array(df)[:,1]
            im = img.imread(sy_loadpath + n + '.jpg')
            # im = np.mean(im, axis=2)
            msdict = msXYZgen(im=im, marker_x=marker_x, marker_y=marker_y)
            
            # np.array(msdict['X']).shape
            
            with open(psave, 'wb') as file:
                pickle.dump(msdict, file)
                print(psave, '저장되었습니다.')
     
    # dictionary에 추가
    for n_num in tqdm(range(len(nlist))):
        n = nlist[n_num]  
        df = pd.read_csv(sy_loadpath + n + '.csv')
        marker_x = np.array(df)[:,0]
        marker_y = np.array(df)[:,1]
        im = img.imread(sy_loadpath + n + '.jpg')
    
        msdict = {'imread':im, 'marker_x':marker_x, 'marker_y':marker_y, 'polygon':None, 'points':None}
        dictionary[n] = msdict

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

#%% dataset1 + 2 // cv 3
def datafile_find(xyz_loadpath):
    flist = os.listdir(xyz_loadpath)
    nlist = []
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.pickle' and flist[i][:7] == 'data_52':
            nlist.append(flist[i])
    nlist = list(set(nlist))
    return nlist

dataset1 = list(datafile_find(mainpath))
dataset2 = list(datafile_find(mainpath_dataset2))
dataset3 = list(datafile_find(mainpath_dataset3))
dataset3.sort()
nlist = datafile_find(mainpath) + dataset2 + dataset3
    
print(len(datafile_find(mainpath)), len(dataset2), len(dataset3))
cvlist = list(dataset3)

#%% model training for dataset3
lnn = 7
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '_dataset3\\'; msFunction.createFolder(weight_savepath)
cv = 0

def path_find(N, dataset1=dataset1, dataset2=dataset2, dataset3=dataset3):
    xyz_loadpath = None
    if nlist[N] in dataset1: xyz_loadpath = mainpath
    elif nlist[N] in dataset2: xyz_loadpath = mainpath_dataset2
    elif nlist[N] in dataset3: xyz_loadpath = mainpath_dataset3
    return xyz_loadpath


for cv in range(len(cvlist)):
# for cv in range(0, len(cvlist)):
    # 1. weight
    
    msid = str(dataset3[cv][15:-7])
    print('cv', cv, msid)
    weight_savename = 'cv_msid_' + msid + '_total_final.h5'
    
    final_weightsave = weight_savepath + weight_savename

    if not(os.path.isfile(final_weightsave)) or False:
        tlist = list(range(len(nlist)))
        telist = np.where(np.array(nlist) == cvlist[cv])[0]
        trlist = list(set(tlist) - set(telist))

        X_te, Y_te, Z_te = [], [] ,[]
        for te in telist:
            gc.collect(); tf.keras.backend.clear_session()
            xyz_loadpath = path_find(te)
            psave = xyz_loadpath + nlist[te]
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                X_te += msdict['X']
                Y_te += msdict['Y']
                Z_te += msdict['Z']
        X_te = np.array(X_te); Y_te = np.array(Y_te); Z_te = np.array(Z_te)
        print(X_te.shape); print(Y_te.shape); print(np.mean(X_te[0]))
        
        if False:
            for ms in [10, 1000,2000,5000]:
                plt.figure()
                plt.imshow(X_te[ms])
        
        model = model_setup(xs=(28, 28, 3), ys=2, lnn=lnn)
        if cv==0: print(model.summary())
        
        epochs = 4; cnt=0
        for epoch in range(epochs):
            print('epoch', epoch)
            for n_num in trlist:
                import time; start = time.time()
                
                gc.collect(); tf.keras.backend.clear_session()
                xyz_loadpath = path_find(n_num)
                psave = xyz_loadpath + nlist[n_num]
                with open(psave, 'rb') as file:
                    msdict = pickle.load(file)
                    X_tmp = np.array(msdict['X'])
                    Y_tmp = np.array(msdict['Y'])
                    Z_tmp = msdict['Z']
                    
                print(n_num, X_tmp.shape, np.mean(Y_tmp, axis=0), np.sum(Y_tmp, axis=0))
                
                hist = model.fit(X_tmp, Y_tmp, epochs=1, verbose=1, batch_size = 2**6)
                print("time :", time.time() - start)
                
                cnt+=1
                if cnt == 20 and False:
                    cell = np.where(Y_tmp[:,1]==1)[0]
                    yhat_cell = model.predict(X_tmp[cell], verbose=1, batch_size = 2**6)
                    print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                    noncell = np.where(Y_tmp[:,1]==0)[0]
                    yhat_noncell = model.predict(X_tmp[noncell], verbose=0, batch_size = 2**6)
                    print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                    cnt = 0
                    cell = np.where(Y_te[:,1]==1)[0]
                    yhat_cell = model.predict(X_te[cell], verbose=1, batch_size = 2**6)
                    print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                    noncell = np.where(Y_te[:,1]==0)[0]
                    yhat_noncell = model.predict(X_te[noncell], verbose=0, batch_size = 2**6)
                    print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                    # print(msFunction.msROC(yhat_noncell[:,1], yhat_cell[:,1]))
                    
            print('---evaluate---')
            model.evaluate(X_te, Y_te, verbose=1, batch_size = 2**6)
            gc.collect()
            tf.keras.backend.clear_session()
        model.save_weights(final_weightsave)

#%% test for dataset3

for cv in range(len(cvlist)):
# for cv in range(0, len(cvlist)):
    # 1. weight
    print('cv', cv)
    msid = str(dataset3[cv][15:-7])
    weight_savename = 'cv_msid_' + msid + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename
    
    if os.path.isfile(final_weightsave):
    
        print(msid)
        psave = weight_savepath + 'sample_n_' + msid + '.pickle'
        t_im = dictionary[msid]['imread']
        # t_im = np.mean(t_im, axis=2)
        marker_x = dictionary[msid]['marker_x']
        marker_y = dictionary[msid]['marker_y']
        positive_indexs = np.transpose(np.array([marker_y, marker_x]))
        polygon = dictionary[msid]['polygon']
        points = dictionary[msid]['points']
    
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
                                for ch in range(3):
                                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                                    # plt.imshow(crop[:,:,ch])
                                X_total_te.append(np.array(crop))
                                Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
                        else:
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

#%% dataset1 + 2 + 3
def datafile_find(xyz_loadpath):
    flist = os.listdir(xyz_loadpath)
    nlist = []
    for i in range(len(flist)):
        if os.path.splitext(flist[i])[1] == '.pickle' and flist[i][:7] == 'data_52':
            nlist.append(flist[i])
    nlist = list(set(nlist))
    return nlist


nlist = datafile_find(mainpath) + \
    datafile_find(mainpath_dataset2) + datafile_find(mainpath_dataset3)
        

print(len(datafile_find(mainpath)), len(datafile_find(mainpath_dataset2)), \
                                        len(datafile_find(mainpath_dataset3)))

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
    
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '\\'; msFunction.createFolder(weight_savepath)
weight_savepath = mainpath + 'weightsave_' + str(lnn) + '_merge\\'; msFunction.createFolder(weight_savepath)

if False:
    df = pd.DataFrame(nlist)
    df = pd.concat((df, pd.DataFrame(cvlist)))
    df.to_csv('C:\\Temp\\20220919_dataset123_develop2\\nlist.csv')          

#%% dataset2

flist2 = os.listdir(mainpath_dataset2)
nlist2 = []
for i in range(len(flist2)):
    if os.path.splitext(flist2[i])[1] == '.pickle' and flist2[i][:7] == 'data_52':
        nlist2.append(flist2[i])
nlist2 = list(set(nlist2))
print(nlist2)
print(len(nlist2))

#%% total training

if False:
    X_te, Y_te, Z_te = [], [] ,[]
    for te in range(5):
        gc.collect(); tf.keras.backend.clear_session()
        psave = mainpath_dataset2 + nlist2[te]
        with open(psave, 'rb') as file:
            msdict = pickle.load(file)
            X_te += msdict['X']
            Y_te += msdict['Y']
            Z_te += msdict['Z']
    X_te = np.array(X_te); Y_te = np.array(Y_te); Z_te = np.array(Z_te)
    print(X_te.shape); print(Y_te.shape); print(np.mean(X_te[0]))

if False: # tmp 임시 model 생성용
    weight_savename = 'merge_20221118.h5'
    final_weightsave = weight_savepath + weight_savename
    
    model = model_setup(xs=(28, 28, 3), ys=2, lnn=7)
    print(model.summary())
    print(nlist)

    epochs = 4; cnt=0
    for epoch in range(epochs):
        for n_num in range(len(nlist)):
            gc.collect(); tf.keras.backend.clear_session()
            psave = xyz_loadpath + nlist[n_num]
            if not(os.path.isfile(psave)): psave = xyz_loadpath + 'dataset2_SORA\\' +  nlist[n_num]
            if not(os.path.isfile(psave)): psave = xyz_loadpath + 'dataset2_sy\\' +  nlist[n_num]
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                X_tmp = np.array(msdict['X'])
                Y_tmp = np.array(msdict['Y'])
                Z_tmp = msdict['Z']
                
            print(n_num, X_tmp.shape, np.mean(Y_tmp, axis=0))
            hist = model.fit(X_tmp, Y_tmp, epochs=1, verbose=1, batch_size = 2**6)
  
        # model.evaluate(X_te, Y_te, verbose=1, batch_size = 2**6)
        model.save_weights(final_weightsave)
    model.save_weights(final_weightsave)
    print('save done')

import sys; sys.exit()

#%% log plot
# import keras
# class BCP(keras.callbacks.Callback):
#     batch_f1_score = [] # accuracy at given batch
#     batch_loss = [] # loss at given batch    
#     def __init__(self):
#         super(BCP,self).__init__() 
#     def on_train_batch_end(self, batch, logs=None):                
#         BCP.batch_f1_score.append(logs.get('f1_score'))
#         BCP.batch_loss.append(logs.get('loss'))
#         # print(logs)

# cvnum = 5
psave = weight_savepath + 'BCP_save.pickle'
if os.path.isfile(psave):
    print('load psave')
    with open(psave, 'rb') as file:
        BCP_save = pickle.load(file)
else:
    BCP_save = msFunction.msarray([cvnum, 4])


lowlist1 = ['C2_1R', 's210302_2R', 's210405_4R', 's210302_4L', 's210225_1R']
lowlist2 = ['C2_2R', 's210225_3L', 's210302_3R', 's210226_3R', 's210226_3L']
len(list(set(lowlist1+lowlist2)))

nlist_idonly = []
for i in range(len(nlist)):
    nlist_idonly.append(nlist[i][15:-7])
    
cvtmp1 = []
for i in lowlist1:
    cvtmp1.append(np.where(np.array(nlist_idonly)==i)[0][0])
cvtmp2 = []
for i in lowlist2:
    cvtmp2.append(np.where(np.array(nlist_idonly)==i)[0][0])   



cvlist_tmp = [[],[],[],[],[], cvtmp1, cvtmp2]

cv = 0
for cv in range(0, len(cvlist_tmp)): # len(cvlist)):
    if len(BCP_save[cv][0]) == 0:
        tlist = list(range(len(nlist)))
        telist = cvlist_tmp[cv]
        trlist = list(set(tlist) - set(telist))

        print(cv, telist)

        X_te, Y_te, Z_te = [], [] ,[]
        for te in telist:
            gc.collect(); tf.keras.backend.clear_session()
            psave = xyz_loadpath + nlist[te]
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                X_te += msdict['X']
                Y_te += msdict['Y']
                Z_te += msdict['Z']
        X_te = np.array(X_te); Y_te = np.array(Y_te); Z_te = np.array(Z_te)
        print(X_te.shape); print(Y_te.shape); print(np.mean(X_te[0]))
        
        
        model = model_setup(xs=(28, 28, 3), ys=2, lnn=lnn)
        if cv==0: print(model.summary())

        epochs=4; cnt=0
        for epoch in range(epochs):
            for n_num in trlist:
                gc.collect(); tf.keras.backend.clear_session()
                psave = xyz_loadpath + nlist[n_num]
                with open(psave, 'rb') as file:
                    msdict = pickle.load(file)
                    X_tmp = np.array(msdict['X'])
                    Y_tmp = np.array(msdict['Y'])
                    Z_tmp = msdict['Z']
                    
                print(n_num, X_tmp.shape, np.mean(Y_tmp, axis=0))
                hist = model.fit(X_tmp, Y_tmp, epochs=1, verbose=1, batch_size = 2**6)

                BCP_save[cv][0].append(hist.history['loss'])
                BCP_save[cv][1].append(hist.history['f1_score'])
                
                hist_te = model.evaluate(X_te, Y_te)
                
                BCP_save[cv][2].append(hist_te[0])
                BCP_save[cv][3].append(hist_te[1][0])
                # 동기화 해야함
                    

with open(psave, 'wb') as f:
    pickle.dump(BCP_save, f, pickle.HIGHEST_PROTOCOL)
    print(psave, '저장되었습니다')
            
#%%
# cv 누적하고, cv끼리 평균 + errorbar로 그래프 완성
mssave7 = msFunction.msarray([4])
for cv in range(7):
    if cv == 4: continue
    tn = len(BCP_save[cv][0])
    mean_recal_loss = np.zeros(tn) * np.nan
    mean_recal_f1 = np.zeros(tn) * np.nan
    
    forlist = list(range(0, tn, int(tn/epochs)))
    forlist += [tn]
    f = forlist[0]
    for i2 in range(len(forlist[:-1])):
        for i in range(forlist[i2], forlist[i2+1]):
            # print(forlist[i2], i)
            mean_recal_loss[i] = np.mean(BCP_save[cv][0][forlist[i2]:i+1])
            mean_recal_f1[i] = np.mean(BCP_save[cv][1][forlist[i2]:i+1])

    mssave7[0].append(mean_recal_loss[:364])
    mssave7[1].append(BCP_save[cv][2][:364])
    
    mssave7[2].append(mean_recal_f1[:364])
    mssave7[3].append(BCP_save[cv][3][:364])
    
plt.figure()
plt.plot(np.mean(np.array(mssave7[0]), axis=0))
plt.plot(np.mean(np.array(mssave7[1]), axis=0))

plt.figure()
plt.plot(np.mean(np.array(mssave7[2]), axis=0))
plt.plot(np.mean(np.array(mssave7[3]), axis=0))

import scipy.stats as stats
plt.figure()
msplot = np.array(mssave7[0])
msplot_mean = np.nanmean(msplot, axis=0)
e = stats.sem(msplot, axis=0, nan_policy='omit')
xaxis = range(len(msplot_mean))
plt.plot(xaxis, msplot_mean, label='Training set')
plt.fill_between(xaxis, msplot_mean+e, msplot_mean-e, alpha=0.5)

msplot = np.array(mssave7[1])
msplot_mean = np.nanmean(msplot, axis=0)
e = stats.sem(msplot, axis=0, nan_policy='omit')
xaxis = range(len(msplot_mean))
plt.plot(xaxis, msplot_mean, label='Test set')
plt.fill_between(xaxis, msplot_mean+e, msplot_mean-e, alpha=0.5)

plt.xlabel('Batchs')
plt.ylabel('Loss')
plt.legend()

figsavepath = 'C:\\SynologyDrive\\study\\dy\\Frontiers in Neuroanatomy\\revision_files\\'
plt.savefig(figsavepath, dpi=200)

#%% 숫자 체크
mssave_num = []
for n_num in trlist:
    gc.collect(); tf.keras.backend.clear_session()
    psave = xyz_loadpath + nlist[n_num]
    with open(psave, 'rb') as file:
        msdict = pickle.load(file)
        X_tmp = np.array(msdict['X'])
        Y_tmp = np.array(msdict['Y'])
        Z_tmp = msdict['Z']

    mssave_num.append([n_num] + list(np.sum(Y_tmp, axis=0)))


np.round(np.mean(np.array(mssave_num)[:,1:], axis=0) * (len(nlist) - 5))
#%% model training

for cv in range(0, 5): # len(cvlist)):
# for cv in range(0, len(cvlist)):
    # 1. weight
    print('cv', cv)
    weight_savename = 'cv_' + str(cv) + '_total_final.h5'
    final_weightsave = weight_savepath + weight_savename

    if not(os.path.isfile(final_weightsave)) or False:
        tlist = list(range(len(nlist)))
        telist = cvlist[cv]
        trlist = list(set(tlist) - set(telist))

        X_te, Y_te, Z_te = [], [] ,[]
        for te in telist:
            gc.collect(); tf.keras.backend.clear_session()
            psave = xyz_loadpath + nlist[te]
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                X_te += msdict['X']
                Y_te += msdict['Y']
                Z_te += msdict['Z']
        X_te = np.array(X_te); Y_te = np.array(Y_te); Z_te = np.array(Z_te)
        print(X_te.shape); print(Y_te.shape); print(np.mean(X_te[0]))
        
        if False:
            for ms in [10, 1000,2000,5000]:
                plt.figure()
                plt.imshow(X_te[ms])
        
        model = model_setup(xs=(29, 29, 3), ys=2, lnn=lnn)
        if cv==0: print(model.summary())
        
        epochs = 4; cnt=0
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
                
                hist = model.fit(X_tmp, Y_tmp, epochs=1, verbose=1, batch_size = 2**6)
                print("time :", time.time() - start)
                
                cnt+=1
                if cnt == 20 and False:
                    cell = np.where(Y_tmp[:,1]==1)[0]
                    yhat_cell = model.predict(X_tmp[cell], verbose=1, batch_size = 2**6)
                    print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                    noncell = np.where(Y_tmp[:,1]==0)[0]
                    yhat_noncell = model.predict(X_tmp[noncell], verbose=0, batch_size = 2**6)
                    print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                    cnt = 0
                    cell = np.where(Y_te[:,1]==1)[0]
                    yhat_cell = model.predict(X_te[cell], verbose=1, batch_size = 2**6)
                    print('cell', np.mean(yhat_cell[:,1]>0.5))
                    
                    noncell = np.where(Y_te[:,1]==0)[0]
                    yhat_noncell = model.predict(X_te[noncell], verbose=0, batch_size = 2**6)
                    print('noncell', np.mean(yhat_noncell[:,1]>0.5))
                    
                    # print(msFunction.msROC(yhat_noncell[:,1], yhat_cell[:,1]))
                    
            model.evaluate(X_te, Y_te, verbose=1, batch_size = 2**6)
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
for cv in range(5): # range(len(cvlist)):
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
                                for ch in range(3):
                                    crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                                    # plt.imshow(crop[:,:,ch])
                                X_total_te.append(np.array(crop))
                                Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
                        else:
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

csv_savepath = 'C:\\Temp\\20220919_dataset123_develop2\\merge_save.csv'
df = pd.read_csv(csv_savepath)
merge_save = np.array(df)[:,-1]
print('len(merge_save)', len(merge_save))
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
    
    if c1 and c2 and (msid in merge_save):
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
        
        tmp = [msid, len(msdict['co']), predicted_cell_n, F1_score, msdict['tp'], msdict['fp'], msdict['fn']]
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

#%% new datasets generalization test

weight_savename = 'dataset1_total.h5'
# weight_savename = 'cv_' + str(cv) + '_total_final.h5'
final_weightsave = weight_savepath + weight_savename

print(nlist2)
print(); print('len(nlist2)', len(nlist2))

n_num = 0
for n_num in range(len(nlist2)):
    test_image_no = nlist2[n_num][15:-7]
    # test_image_no = 'bvPLA2 1-1.JPG_resize_crop'
    
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
                            for ch in range(3):
                                crop[:,:,ch] = crop[:,:,ch]/np.mean(crop[:,:,ch])
                                # plt.imshow(crop[:,:,ch])
                            X_total_te.append(np.array(crop))
                            Z_total_te.append([row-SIZE_HF, col-SIZE_HF])
                    else:
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

#%%

def F1_score_optimization(nlist=None, dictionary=dictionary, get_F1=get_F1):
    mssave9 = []
    nlist_for = list(nlist)
    for n_num in range(len(nlist_for)):
        print('F1 score optimization', n_num)
        test_image_no = nlist_for[n_num][15:-7]
        # test_image_no = '5.tif_resize_crop'
        psave = weight_savepath + 'sample_n_' + str(test_image_no) + '.pickle'
        psave_f1parameters = weight_savepath + 'F1_parameters_' + str(test_image_no) + '.pickle'
        
        if os.path.isfile(psave):
            with open(psave, 'rb') as file:
                msdict = pickle.load(file)
                yhat_save = msdict['yhat_save']
                z_save = msdict['z_save']
                
            t_im = dictionary[test_image_no]['imread']
            t_im = np.mean(t_im, axis=2) # only for vis
            marker_x = dictionary[test_image_no]['marker_x']
            marker_y = dictionary[test_image_no]['marker_y']
            positive_indexs = np.transpose(np.array([marker_y, marker_x]))
            polygon = dictionary[test_image_no]['polygon']
        
        
            if not(os.path.isfile(psave_f1parameters)):
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
                    contour_thr_list = list(range(20,300,1))
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
                
                tresholds_list = np.round(np.arange(0.5,0.99,0.01), 3)
                forlist = list(tresholds_list)
                pre_F1 = -1
                for forlist_cpu in forlist:
                    mssave = ray_F1score_cal([forlist_cpu], yhat_save=yhat_save, positive_indexs=positive_indexs, \
                                                             t_im=t_im, polygon=polygon)
                    output_ids.append(mssave[0])
                    if mssave[0][-1] < pre_F1: break
                    pre_F1 = mssave[0][-1]
                    
                tresholds_list = np.round(np.arange(0.49,0,-0.01), 3)
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
                
                if not(os.path.isfile(psave_f1parameters)) or True:
                    with open(psave_f1parameters, 'wb') as f:  # Python 3: open(..., 'rb')
                        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                        print(psave_f1parameters, '저장되었습니다.')
                gc.collect()
                t1 = time.time() - start; print('\n', "time F1 optimize :", t1)  # 현재시각 - 시작시간 = 실행 시간

        #% report는 따로 있을 필요가 없는듯 하니 여기에 병합
            with open(psave_f1parameters, 'rb') as file:
                result = pickle.load(file)
                
            threshold = result[2]
            contour_thr = result[3]
            
            F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
                       yhat_save=yhat_save, positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
                
            predicted_cell_n = len(msdict['los'])
            
            tmp = [test_image_no, len(msdict['co']), predicted_cell_n, F1_score, \
                   msdict['tp'], msdict['fp'], msdict['fn'], msdict['predbox2'], msdict['ansbox']]
            mssave9.append(tmp)
            print('reports >', tmp[:-2])
            
    return mssave9

#%%

mssave2 = F1_score_optimization(nlist=nlist, dictionary=dictionary)
mssave2 = np.array(mssave2)
print()
print('F1 score mean', np.mean(np.array(mssave2[:,3], dtype=float)))


mssave2 = F1_score_optimization(nlist=nlist2, dictionary=dictionary)
mssave2 = np.array(mssave2)
print()
print('F1 score mean', np.mean(np.array(mssave2[:,3], dtype=float)))

mssave2 = F1_score_optimization(nlist=dataset3, dictionary=dictionary)
mssave3 = np.array(mssave2)[:,:-2]
len(mssave3)
print()
tmp = np.array(mssave3[:,3], dtype=float)
vix = np.where(tmp > 0.54)[0]
print('F1 score mean', np.mean(tmp[vix]))

#%%

ix = 4
mssave2[ix][0]
plt.imshow(mssave2[ix][-2])
plt.imshow(mssave2[ix][-1])








































