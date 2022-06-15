# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:07:08 2022

@author: PC
"""

import tensorflow as tf
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
#%% rawdata_dict gen

# img load, id list gen
img_list = []
id_list = []
path = 'C:\\SynologyDrive\\study\\dy\\52\\imgs\\'
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
path = 'C:\\SynologyDrive\\study\\dy\\52\\markers\\'
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
path = 'C:\\SynologyDrive\\study\\dy\\52\\roiBorder\\'
for (path, dir, files) in os.walk(path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.csv':
            print("%s/%s" % (path, filename))
            
            filepath = path + '//' + filename
            msid = filename[10:-4]
            if not(msid in id_list[:,0]): print(filename, 'missing')
            ROI_list.append([msid, filepath])
ROI_list = np.array(ROI_list)

# gen dict from key(=id)

dic = {}
for i in tqdm(range(len(id_list))):
    msid = id_list[i][0]
    
    idix = np.where(img_list[:,0] == msid)[0][0]
    im = img.imread(img_list[idix, 1])
    image = Image.open(img_list[idix, 1])
    width = image.size[0]
    length = image.size[1]
    
    idix = np.where(marker_list[:,0] == msid)[0][0]
    df = np.array(pd.read_csv(marker_list[idix, 1]))
    marker_x = df[:,2]
    marker_y = df[:,3]
    
    idix = np.where(marker_list[:,0] == msid)[0][0]
    df = np.array(pd.read_csv(marker_list[idix, 1]))
    marker_x = np.array(df[:,2], dtype=int)
    marker_y = np.array(df[:,3], dtype=int)

    r_dict = {'marker_x':marker_x,
              'marker_y':marker_y,
              'imread': im,
              'width': width,
              'length': length}
    
    dic[msid] = r_dict

psave = 'C:\\SynologyDrive\\study\\dy\\52\\' +'data_52_ms.pickle'
with open(psave, 'wb') as file:
    pickle.dump(dic, file)
    print(psave, '저장되었습니다.')



#%% """HYPERPARAMETERS"""
mainpath = 'C:\\SynologyDrive\\study\\dy\\48\\'
nct = 50 # INTENSITY 조정값
B = 0.5 # NEGATIVE CROP INT 적용 CROP 비율

#positive crop 확장--------------
expansion = 1 # 변경 시 line 84 부근 수정 필요

# predict threshold 변경--------
threshold = 0.9

#crop 크기 수정------------------                                    
size_hf = 14 # crop size: 29 x 29 

# cell 사이즈 검색 범위
size1 = 1
size2 = 120

#데이터 추가시 수정필요한 부분----
first_file_no = 101                                                        
last_file_no = 305


#########변수이름변경###########
sh = size_hf
size = (size_hf*2)+1
eps = expansion
###############################

dic = {}
#%%

def chop (a):
    if a % 2 == 0: b = [int(a/2),int(a/2)]
    else: b = [int(a/2), int(a/2)+1]
    return b

# 크롭 이미지 좌표 범위의 최솟값 결정
def find_min (t, u):
    if t-u<=0: return 0
    elif t-u>=0: return t-u
    else: return None
    
def find_max (t, u, total):
    if t+u>=total:
        return total
    elif t+u<=total:
        return t+u
    else:
        return None
    
def find_square (y=None, x=None, unit=None, \
                 im=None, chop=chop, find_min=find_min, find_max=find_max, \
                 height=None, width=None):
    
    import numpy as np
    square = im[np.max([0, y-unit]):np.min([height, y+unit]), np.max([0, x-unit]):np.min([width, x+unit])]
    square = square / np.mean(square)
    # plt.imshow(square)
    extend_x = size - find_max(x, unit, width) + find_min(x,unit)
    extend_y = size - find_max(y, unit, height) + find_min(y,unit)
    padding= np.lib.pad(square,((chop(extend_y)[0],chop(extend_y)[1]), (chop(extend_x)[0],chop(extend_x)[1]), (0,0)),'constant', constant_values=(255))
    return padding

def im_mean(crop):
    import cv2
    import numpy as np
    
    piece = (crop).astype(np.uint8)
    piece =cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
    piece=cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    p = cv2.mean(piece)
    return p[0]

def im_aug(crop, yin, zin):
    import numpy as np
    # from PIL import Image,ImageFilter, ImageEnhance, ImageOps
    
    new = Image.fromarray((crop).astype(np.uint8))
    
    bu=ImageEnhance.Brightness(new).enhance(1.2)
    bucu = ImageEnhance.Contrast(bu).enhance(1.2)
    bucd = ImageEnhance.Contrast(bu).enhance(0.8)
    bu = np.array(bu) ; bucu = np.array(bucu) ; bucd = np.array(bucd)
    
    bd = ImageEnhance.Brightness(new).enhance(0.8)
    bdcu = ImageEnhance.Contrast(bd).enhance(1.2)
    bdcd = ImageEnhance.Contrast(bd).enhance(0.8)
    bd = np.array(bd) ; bdcu = np.array(bdcu) ;  bdcd = np.array(bdcd)
    
    cu = ImageEnhance.Contrast(new).enhance(1.2)
    cu = np.array(cu)
    cd = ImageEnhance.Contrast(new).enhance(0.8)
    cd = np.array(cd)
    
    xout = np.array([bu, bd, cu, cd, bucu, bucd, bdcu, bdcd])
    
    # y
    yout = np.zeros((len(xout),2))
    yout[:, np.argmax(yin)] = 1
    
    # z
    zout = np.zeros((len(xout),len(zin)))
    zout[:, :len(zin)] = zin
    
    return xout, yout, zout

def get_F1(threshold=None, contour_thr=None,\
           height=None, width=None, yhat_save=None, \
               z_save=None, t_im=None, positive_indexs=None, polygon=None):

    import numpy as np
    # import cv2
    
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
        if M['m00'] < contour_thr: 
            pass
        elif M['m00'] > 500:
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # dict2 체크후 사용
            if not(polygon is None):
                code = Point(cx,cy)
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
        if not(polygon is None):
            code = Point(col, row)
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
                
    
    if False:
        az = np.zeros((400*int(len(co)),2))*np.nan
        for z in range(len(co)):
            zx = co[z][0]
            zy = co[z][1]
            for zz in range(0,20):
                az[z*400+(20*zz):z*400+zz*20+20,0] = list(range(zx-10,zx+10))
                az[z*400+zz*20:z*400+zz*20+20,1] = list(range(zy-10,zy+10))[zz]
    
        white2 = np.zeros((height,width,3))*np.nan
        white2[:,:] = [255,255,255]
        for i in range(len(az)):
            white2[int(az[i][1]),int(az[i][0])] = [0,0,0]
            
    else:
        dots = co
        white2 = dot_expand(height=height, width=width, dots=dots)
            
    # plt.imshow(white2)    # ground truth area
    tparea = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        tparea.append((w[1][i],w[0][i]))     

    # predicted cells area
    if False:
        pur = np.zeros((400*int(len(pink)),2))*np.nan
        for z in range(len(pink)):
            zx = pink[z][0]
            zy = pink[z][1]
            for zz in range(0,20):
                pur[z*400+(20*zz):z*400+zz*20+20,0] = list(range(zx-10,zx+10))
                pur[z*400+zz*20:z*400+zz*20+20,1] = list(range(zy-10,zy+10))[zz]
    
        white4 = np.zeros((height,width,3))*np.nan
        white4[:,:] = [255,255,255]
        for i in range(len(pur)):
            white4[int(int(pur[i][1])),int(pur[i][0])] = [0,0,0]
    else:
        dots = pink
        white4 = dot_expand(height=height, width=width, dots=dots)
        
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
    fp = len(pink)
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
    
    msdict = {'los': los, 'white': white}
    
    return F1_score, msdict

#%% keras setup
def model_setup(xs=None, ys=None, lr=1e-4):
    # import tensorflow as tf
    # from tensorflow.keras import datasets, layers, models, regularizers
    # from tensorflow.keras.layers import BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    model = models.Sequential()
    model.add(layers.Conv2D(2**11, (4, 4), activation='relu', input_shape=xs))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**11, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**11, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # model.add(layers.Conv2D(2**8, (5, 5), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='relu' ))
    model.add(layers.Dense(2**7, activation='sigmoid') )
    model.add(layers.Dense(ys, activation='softmax'))

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               # loss='binary_crossentropy',

    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(learning_rate=lr, decay=1e-3, beta_1=0.9, beta_2=0.999), \
                  loss='categorical_crossentropy', metrics=['accuracy']) 
    
    return model

model = model_setup(xs=(29,29,3), ys=(2))
print(model.summary())

#%%
d = 25; gap=[]
for row in range(-d,d):
    for col in range(-d,d):
        distance = np.sqrt(row**2 + col**2)
        if distance <= d:
            gap.append((row,col))

def gen_total_index(height=None, width=None, polygon=None, retangle_roi_dict=None):
    from shapely.geometry import Point
    
    rowmin = retangle_roi_dict['rowmin']
    rowmax = retangle_roi_dict['rowmax']
    colmin = retangle_roi_dict['colmin']
    colmax = retangle_roi_dict['colmax']

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

# total_index = gen_total_index(height=768, width=1254, polygon=None)
# total_index_save = {'768_1254' : total_index}
#%% XYZgen """train X, Y 만들기"""

# ms_filepath = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'data_48.pickle'
# with open(ms_filepath, 'rb') as file:
#     dictionary_old = pickle.load(file)

# ms_filepath2 = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'roipoints_48.pickle'
# with open(ms_filepath2, 'rb') as file:
#     dictionary2_old = pickle.load(file)

# ms_filepath = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'data_51.pickle'
# with open(ms_filepath, 'rb') as file:
#     dictionary = pickle.load(file)

# ms_filepath2 = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'roipoints_51.pickle'
# with open(ms_filepath2, 'rb') as file:
#     dictionary2 = pickle.load(file)

# keylist = list(dictionary.keys())
# keylist2 = list(dictionary2.keys())

# #%% key 매칭 old - new
# for n_num in range(len(keylist)):
#     n = keylist[n_num]
#     marker_x = dictionary[n]['marker_x']
    
#     for i in range(len(dictionary_old)):
#         oldkey = list(dictionary_old.keys())
#         marker_x_old = dictionary_old[oldkey[i]]['marker_x']
#         if len(marker_x_old) == len(marker_x):
#             if np.sum(np.abs(np.array(marker_x_old) - np.array(marker_x)))==0:
#                 print(n_num, oldkey[i])
#                 dictionary[n]['oldkey'] = oldkey[i]

# psave = 'C:\\SynologyDrive\\study\\dy\\52\\' +'data_52_ms.pickle'
# with open(psave, 'wb') as file:
#     pickle.dump(dic, file)
#     print(psave, '저장되었습니다.')
    
ms_filepath2 = 'C:\\SynologyDrive\\study\\dy\\52\\' +'data_52_ms.pickle'
with open(ms_filepath2, 'rb') as file:
    dictionary = pickle.load(file)

keylist = list(dictionary.keys())
#%%
prev_len=0
total_len = 0


psave = 'C:\\SynologyDrive\\study\\dy\\52\\' +'data_52_ms_XYZ.pickle'
if not(os.path.isfile(psave)):

    X, Y, Z = [], [], []
    for n_num in tqdm(range(len(keylist))):
        n = keylist[n_num]
    
        marker_x = dictionary[n]['marker_x']
        marker_y = dictionary[n]['marker_y']
        width = dictionary[n]['width']
        height = dictionary[n]['length']
        im = dictionary[n]['imread']
        
        # try:
        #     polygon = dictionary2[n]['polygon']
        #     points = dictionary2[n]['points']
        # except:
        polygon = None
        points = None
        
        # retangle ROI
        rowmin = np.max([np.min(marker_y) - 50, 0])
        rowmax = np.min([np.max(marker_y) + 50, height])
        colmin = np.max([np.min(marker_x) - 50, 0])
        colmax = np.min([np.max(marker_x) + 50, width])
        retangle_roi_dict = {'rowmin': rowmin, 'rowmax': rowmax, 'colmin': colmin, 'colmax': colmax}
        
        if False:
            im2 = np.array(im, dtype=np.uint8)
            r = 10
            for k in range(len(points)):
                x = points[k][0]
                y = points[k][1]
                im2[y-r:y+r, x-r:x+r :] = 0       
            plt.imshow(im2)
        
        # total index
        # mskey = str(height) + '_' + str(width)
        # if mskey in list(total_index_save.keys()):
        #     total_index = total_index_save[mskey]
        # else:
        total_index = gen_total_index(height=height, width=width, \
                                      polygon=polygon, retangle_roi_dict=retangle_roi_dict)
        
        int_value = []
        # positive crop
        positive_index = [] 
        for i in range(len(marker_x)):
            crop = find_square(y=marker_y[i], x=marker_x[i], unit=size_hf, im=im, height=height, width=width)
            
            # crop = crop / np.mean(crop)
            X.append(crop)
            Y.append([0,1])
            Z.append([n, marker_y[i], marker_x[i]])
            positive_index.append((marker_y[i], marker_x[i]))
    
            int_value.append(im_mean(crop))
            
        int_thr = int(round(np.mean(int_value)))
        
        gap_index = []
        for n1 in range(len(positive_index)):
            for n2 in range(len(gap)):
                row = positive_index[n1][0] + gap[n2][0]
                col = positive_index[n1][1] + gap[n2][1]
                gap_index.append((row,col))
                
        negative_index = list(set(total_index) - set(positive_index) - set(gap_index))
        negative_n = len(negative_index)
        
        # negative crop
        negative_cnt = 0
        # occupied = np.transpose(np.array([marker_x, marker_y]))
        
        # negative_n/len(positive_index)
        
        epochs = 1000000000
        for epoch in range(epochs):
            if negative_cnt > len(marker_x) * 20: break
        
            rix = random.randint(0, negative_n-1)
            row = negative_index[rix][0]
            col = negative_index[rix][1]
    
            pc = find_square(y=row, x=col, unit=size_hf, im=im, height=height, width=width)
            
            neg_int = im_mean(pc)
           
            # condition 2
            if neg_int < int_thr + nct:
                 negative_cnt += 1
                 X.append(pc)
                 Y.append([1,0])
                 Z.append([n, row, col])
        
        # roi 바깥쪽
        # negative_cnt = 0
        # for epoch in range(epochs):
        #     if negative_cnt > len(marker_x) * 2: break
    
        #     row = random.randint(rowmin, rowmax)
        #     col = random.randint(colmin, colmax)
            
        #     if not (row,col) in gap_index:
        #         passsw = False
        #         if not polygon is None:
        #             code = Point(col,row)
        #             if not(code.within(polygon)): passsw = True
        #         else: passsw = True
        #         if passsw:
        #             pc = find_square(y=row, x=col, unit=size_hf, im=im, height=height, width=width)
        
        #             negative_cnt += 1
        #             X.append(pc)
        #             Y.append([1,0])
        #             Z.append([n, row, col])
            
            # if epochs == epoch: print(n, 'ng shorts')
        
        if True:
            positive = np.where(np.logical_and(np.array(Z)[:,0]==n, np.array(Y)[:,1]==1))[0]
            negative = np.where(np.logical_and(np.array(Z)[:,0]==n, np.array(Y)[:,0]==1))[0]
            
            allo = np.ones((height,width))
            br = 3
            for y in positive:
                allo[np.max([0, Z[y][1]-br]):np.min([height, Z[y][1]+br]), \
                     np.max([0, Z[y][2]-br]):np.min([width, Z[y][2]+br])] = 2
            
            br = 3
            for y in negative:
                allo[np.max([0, Z[y][1]-br]):np.min([height, Z[y][1]+br]), \
                     np.max([0, Z[y][2]-br]):np.min([width, Z[y][2]+br])] = 0
    
            plt.figure()
            plt.imshow(allo, cmap='binary')
            plt.title(n)
            
    print(X.shape, len(X), 'np.mean(Y, axis=0)', np.mean(Y, axis=0))
    # balancing
    plabel = np.where(Y[:,1]==1)[0]
    X_tmp = X[plabel]
    Y_tmp = Y[plabel]
    Z_tmp = Z[plabel]
    for repeat in range(4):
        X_tmp = np.concatenate((X_tmp, X_tmp), axis=0)
        Y_tmp = np.concatenate((Y_tmp, Y_tmp), axis=0)
        Z_tmp = np.concatenate((Z_tmp, Z_tmp), axis=0)
    
    X = np.concatenate((X, X_tmp), axis=0)
    Y = np.concatenate((Y, Y_tmp), axis=0)
    Z = np.concatenate((Z, Z_tmp), axis=0)
        
    print(X.shape, len(X), 'np.mean(Y, axis=0)', np.mean(Y, axis=0))
    
    # shuffle
    rlist = list(range(len(X)))
    random.seed(1)
    random.shuffle(rlist)
    X = X[rlist]
    Y = Y[rlist]
    Z = Z[rlist]
    
    X = np.array(X); Y = np.array(Y); Z = np.array(Z)
    msdict = {'X':X, 'Y':Y, 'Z':Z}
    with open(psave, 'wb') as file:
        pickle.dump(msdict, file)
        print(psave, '저장되었습니다.')
#%% cvset

with open(psave, 'rb') as file:
    msdict = pickle.load(file)
    X = msdict['X']
    Y = msdict['Y']
    Z = msdict['Z']
    
# print(Y[:10,0]   )

# LOSO cv
tlist = list(range(len(X)))
cvlist = []
# print(Z_vix[:,3])
session_list = list(set(np.array(Z[:,0])))
session_list.sort()
# print(session_list[:10])

for se in range(len(session_list)):
    msid = session_list[se]
    telist = np.where(np.array(Z[:,0])==msid)[0]
    print(se, msid, len(cvlist))     
    cvlist.append([telist, msid])

print('len(cvlist)', len(cvlist))

import sys; sys.exit()
#%% cv training
cv = 0; mssave2 = []
print(cvlist[cv][1])
for cv in range(0, len(cvlist)):
    weight_savename = 'cv_' + str(cv) + '_subject_' + str(cvlist[cv][1]) + '_total_final.h5'
    final_weightsave =  'C:\\SynologyDrive\\study\\dy\\52\\weightsave\\' + weight_savename
    
    if not(os.path.isfile(final_weightsave)) or False:
        telist = cvlist[cv][0]
        trlist = list(set(tlist)-set(telist))
        
        X_tr = X[trlist]; Y_tr = Y[trlist]; Z_tr = Z[trlist]
        X_te = X[telist]; Y_te = Y[telist]; Z_te = Z[telist]
        
        print(len(X_tr), len(X_te))
        print(np.mean(Y_tr, axis=0), np.mean(Y_te, axis=0))
        
        # X_aug, Y_aug, Z_aug = [], [], []
        # for i in tqdm(range(len(X_tr))):
        #     xout, yout, zout = im_aug(X_tr[i], Y_tr[i], Z_tr[i])
        #     X_aug += list(xout)
        #     Y_aug += list(yout)
        #     Z_aug += list(zout)
        # X_aug, Y_aug, Z_aug = np.array(X_aug), np.array(Y_aug), np.array(Z_aug)
            
        # X_tr = np.concatenate((X_tr, X_aug), axis=0)
        # Y_tr = np.concatenate((Y_tr, Y_aug), axis=0)
        # Z_tr = np.concatenate((Z_tr, Z_aug), axis=0)
        
        print(X_tr.shape)
        print(np.mean(Y_tr, axis=0), np.mean(Y_te, axis=0))
#%
        model = model_setup(xs=X_tr[0].shape, ys=Y_tr[0].shape[0])
        model.fit(X_tr, Y_tr, epochs=2, verbose=1, batch_size = 2**3, validation_data=(X_te, Y_te))
        model.save_weights(final_weightsave)

    #%%

    #% test all set
    test_image_no = cvlist[cv][1]
    psave = mainpath + 'sample_n_' + str(test_image_no) + '.pickle'
    model.load_weights(final_weightsave)
    
    width = dictionary[test_image_no]['width']
    height = dictionary[test_image_no]['length']
    t_im = dictionary[test_image_no]['imread']
    marker_x = dictionary[test_image_no]['marker_x']
    marker_y = dictionary[test_image_no]['marker_y']
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    # try:
    #     polygon = dictionary2[test_image_no]['polygon']
    #     points = dictionary2[test_image_no]['points']
    # except:
    polygon = None
    points = None
        
    # if False:
    #     polygon = dictionary2[test_image_no]['polygon']
    #     points = dictionary2[test_image_no]['points']
    
    # retangle ROI
    rowmin = np.max([np.min(marker_y) - 50, 0])
    rowmax = np.min([np.max(marker_y) + 50, height])
    colmin = np.max([np.min(marker_x) - 50, 0])
    colmax = np.min([np.max(marker_x) + 50, width])
    retangle_roi_dict = {'rowmin': rowmin, 'rowmax': rowmax, 'colmin': colmin, 'colmax': colmax}
        
    if False:
        xtmp = []
        vix = np.where(Y_te[:,1]==1)[0]
        for j in vix:
            y = Z_te[j,1]; x = Z_te[j,2]
            unit=sh; im=t_im; height=height; width=width
            crop = find_square(y=y, x=x, unit=unit, im=im, height=height, width=width)
            # print(y, x)
            # plt.imshow(crop)
            # plt.plot(np.mean(crop, axis=0))
            # np.mean(crop)
            xtmp.append(crop)
        xtmp = np.array(xtmp)
        ytmp = model.predict(xtmp)
        print(np.mean(ytmp[:,1]>0.5))

    if not(os.path.isfile(psave)):
        import ray

        
        # allo = np.zeros((t_im.shape[0], t_im.shape[1])) * np.nan
        yhat_save = []
        z_save = []
        
        ##
        import time
        start = time.time()  # 시작 시간 저장
        
        cpus = 14
        ray.shutdown()
        ray.init(num_cpus=cpus)
        @ray.remote
        def test_crop(forlist_cpu, height=None, width=None, sh=None, t_im=None, polygon=None):
            X_total_te_tmp, Z_total_te_tmp = [] ,[]
            for row in forlist_cpu:
                for col in range(colmin, colmax):
                    y=row; x=col; unit=sh; im=t_im; height=height; width=width
                    crop = find_square(y=y, x=x, unit=unit, im=im, height=height, width=width)
                    if crop.shape == (29, 29, 3):
    
                        # crop = crop / np.mean(crop)
                        if not(polygon is None):
                            code = Point(col,row)
                            if code.within(polygon):
                                X_total_te_tmp.append(crop)
                                Z_total_te_tmp.append([row, col])
                        else:
                            X_total_te_tmp.append(crop)
                            Z_total_te_tmp.append([row, col])
            return X_total_te_tmp, Z_total_te_tmp
        
        
        forlist = list(range(rowmin, rowmax))
        div = int(len(forlist)/cpus)
        output_ids = []
        for cpu in range(cpus):
            print(cpu)
            if cpu != cpus-1: forlist_cpu = forlist[cpu*div : (cpu+1)*div]
            elif cpu == cpus-1: forlist_cpu = forlist[cpu*div :]

            output_ids.append(test_crop.remote(forlist_cpu, height=height, width=width, sh=sh, t_im=t_im, polygon=polygon))
        output_list = ray.get(output_ids)
        
        X_total_te = []
        Z_total_te = []
        for cpu in range(cpus):
            X_total_te += output_list[cpu][0]
            Z_total_te += output_list[cpu][1]
            
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        
        ##
        import time
        start = time.time()  # 시작 시간 저장
        
        yhat_save = []
        def ray_keras_predict(X_total_te):
            X_total_te = np.array(X_total_te)
            yhat = model.predict(X_total_te)
            return list(yhat[:,1])
        
        forlist = list(X_total_te)
        div = int(len(forlist)/cpus)
        # output_ids = []
        for cpu in range(cpus):
            print(cpu)
            if cpu != cpus-1: forlist_cpu = forlist[cpu*div : (cpu+1)*div]
            elif cpu == cpus-1: forlist_cpu = forlist[cpu*div :]
            yhat_save += (ray_keras_predict(forlist_cpu))
            
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        
        z_save = list(Z_total_te)
        msdict = {'yhat_save': yhat_save, 'z_save': z_save}
        
        if not(os.path.isfile(psave)) or True:
            with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
                pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                print(psave, '저장되었습니다.')

    with open(psave, 'rb') as file:
        msdict = pickle.load(file)
        yhat_save = msdict['yhat_save']
        z_save = msdict['z_save']
    
    # optimize threshold, contour_thr
    import time
    start = time.time()  # 시작 시간 저장
    
    cpus = 14
    ray.shutdown()
    ray.init(num_cpus=cpus)
    
    @ray.remote
    def ray_F1score_cal(forlist_cpu, height=None, width=None, yhat_save=yhat_save, \
                        positive_indexs=None, z_save=z_save, t_im=None, polygon=None):
        
        mssave = []
        for threshold in forlist_cpu:
            for contour_thr in range(30,100,3):
                F1_score, _ = get_F1(threshold=threshold, contour_thr=contour_thr,\
                            height=height, width=width, yhat_save=yhat_save, \
                                positive_indexs=positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
    
                # F1_score = 1
    
                mssave.append([threshold, contour_thr, F1_score])
        return mssave
    
    tresholds_list = np.round(np.arange(0.1,0.9,0.01), 3)
    forlist = list(tresholds_list)
    div = int(len(forlist)/cpus)
    output_ids = []
    for cpu in range(cpus):
        print(cpu)
        if cpu != cpus-1: forlist_cpu = forlist[cpu*div : (cpu+1)*div]
        elif cpu == cpus-1: forlist_cpu = forlist[cpu*div :]

        output_ids.append(ray_F1score_cal.remote(forlist_cpu, height=height, width=width, \
                                                  positive_indexs=positive_indexs, t_im=t_im, polygon=polygon))
            
    output_list = ray.get(output_ids)
    mssave = []
    for cpu in range(cpus):
        mssave += output_list[cpu]
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    mssave = np.array(mssave)
    mix = np.argmax(mssave[:,2])
    print()
    print('max F1 score', [cv, test_image_no] + list(mssave[mix,:]))
    mssave2.append([cv, test_image_no] + list(mssave[mix,:]))

print(len(mssave2))
print(np.mean(np.array(mssave2)[:,-1]))

#%% 20220609 - ROI 모두 지정 후 재평가
# XYZ 없이, oldkey만 가지고 test result 불러 온 뒤, "optimize threshold, contour_thr" 만 진행 

mssave5 = []
for n_num in range(len(keylist)):
    n = keylist[n_num]
    test_image_no = dictionary[n]['oldkey']
    psave = mainpath + 'sample_n_' + str(test_image_no) + '.pickle'

    with open(psave, 'rb') as file:
        msdict = pickle.load(file)
        yhat_save = msdict['yhat_save']
        z_save = msdict['z_save']
    
    #
    
    width = dictionary[n]['width']
    height = dictionary[n]['length']
    t_im = dictionary[n]['imread']
    marker_x = dictionary[n]['marker_x']
    marker_y = dictionary[n]['marker_y']
    positive_indexs = np.transpose(np.array([marker_y, marker_x]))
    
    polygon = dictionary2[n]['polygon']
    
    #
    psave = mainpath + 'thr_optimized_f1score.pickle'
    if not(os.path.isfile(psave)) or False:
        with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
            pickle.dump(mssave2, f, pickle.HIGHEST_PROTOCOL)
            print(psave, '저장되었습니다.')

    # optimized data load
    with open(psave, 'rb') as file:
        mssave2 = pickle.load(file)
    mssave2 = np.array(mssave2)
    
    #
    
    ix = np.where(mssave2[:,1] == test_image_no)[0][0]
    threshold = mssave2[ix,2]
    contour_thr = mssave2[ix,3]
    
    mssave = []
    

    F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
               height=height, width=width, yhat_save=yhat_save, \
                   positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
        
    # print(n, mssave2[ix,4], '>>' , F1_score)
    
    predicted_cell_n = len(msdict['los'])
    
    #
    print(n, predicted_cell_n, F1_score)
    mssave5.append([n, predicted_cell_n, F1_score])
 

#%% after optimization

# psave = mainpath + 'thr_optimized_f1score.pickle'
# if not(os.path.isfile(psave)) or False:
#     with open(psave, 'wb') as f:  # Python 3: open(..., 'rb')
#         pickle.dump(mssave2, f, pickle.HIGHEST_PROTOCOL)
#         print(psave, '저장되었습니다.')

# # optimized data load
# with open(psave, 'rb') as file:
#     mssave2 = pickle.load(file)
# mssave2 = np.array(mssave2)
#     # [cvnum, imgnum, threshold, contour_thr, optimized_F1score]

# # ms_filepath = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'data_51.pickle'
# # with open(ms_filepath, 'rb') as file:
# #     dictionary = pickle.load(file)

# # ms_filepath2 = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'roipoints_48.pickle'
# # with open(ms_filepath2, 'rb') as file:
# #     dictionary2 = pickle.load(file)
    
# # keylist = list(dictionary.keys())
# # keylist2 = list(dictionary2.keys())

# import pandas as pd
# import sys; 
# sys.path.append('D:\\mscore\\code_lab\\')
# sys.path.append('C:\\mscode')
# import msFunction

# df = np.array(pd.read_excel(mainpath + 'mouse_20220602.xlsx'))

# msGroup = {}
# for i in range(len(df)):
#     mskey = df[i,0]
#     mslabel = df[i,-1]
#     # print(mskey, mslabel)
#     msGroup[mskey] = mslabel

# #%% adhoc analysis

# mssave3 = msFunction.msarray([2])
# mssave4 = []
# for N in range(len(keylist)):
#     test_image_no = keylist[N]
#     if not test_image_no in [31, 'A3.1L']:
#         # load
#         psave = mainpath + 'sample_n_' + str(test_image_no) + '.pickle'
        
#         width = dictionary[test_image_no]['width']
#         height = dictionary[test_image_no]['length']
#         t_im = dictionary[test_image_no]['imread']
#         marker_x = dictionary[test_image_no]['marker_x']
#         marker_y = dictionary[test_image_no]['marker_y']
#         positive_indexs = np.transpose(np.array([marker_y, marker_x]))
#         try:
#             polygon = dictionary2[test_image_no]['polygon']
#             points = dictionary2[test_image_no]['points']
#         except:
#             polygon = None
#             points = None
            
#         if not(polygon is None):
#             with open(psave, 'rb') as file:
#                 msdict = pickle.load(file)
#                 yhat_save = msdict['yhat_save']
#                 z_save = msdict['z_save']
                   
#             # thr load
#             ix = np.where(mssave2[:,1] == test_image_no)[0][0]
#             threshold = mssave2[ix,2]
#             contour_thr = mssave2[ix,3]
            
#             mssave4.append(mssave2[ix,4])
            
#             F1_score, msdict = get_F1(threshold=threshold, contour_thr=contour_thr,\
#                        height=height, width=width, yhat_save=yhat_save, \
#                            positive_indexs= positive_indexs, z_save=z_save, t_im=t_im, polygon=polygon)
            
#             predicted_cell_n = len(msdict['los'])
            
#             #
#             print(N, msGroup[test_image_no])
#             if msGroup[test_image_no] == 'mptp':
#                 mssave3[1].append([test_image_no, predicted_cell_n])
#             elif  msGroup[test_image_no] == 'saline':
#                 mssave3[0].append([test_image_no, predicted_cell_n])

# Aprism = pd.DataFrame(np.array(mssave3[0])[:,1])
# Aprism = pd.concat((Aprism, pd.DataFrame(np.array(mssave3[1])[:,1])), axis=1, ignore_index=True)    

# #%%


# mssave2 = np.array(mssave2)
# mssave3 = msFunction.msarray([2])
# for i in range(len(mssave2)):
#     if msGroup[mssave2[i,1]] == 'mptp':
#         mssave3[1].append([int(mssave2[i,1]), mssave2[i,4]])
#     elif  msGroup[mssave2[i,1]] == 'saline':
#         mssave3[0].append([int(mssave2[i,1]), mssave2[i,4]])
        
                           
# #%%
# mssave3 = msFunction.msarray([2])
# mssave4 = []
# for cv in range(0, len(cvlist)):
#     test_image_no = cvlist[cv][1]
#     image_no = test_image_no
#     marker_x = dictionary[test_image_no]['marker_x']
#     marker_y = dictionary[test_image_no]['marker_y']
    
#     try:
#         polygon = dictionary2[test_image_no]['polygon']
#         points = dictionary2[test_image_no]['points']
#     except:
#         polygon = None
#         points = None

#     if not(polygon is None):
#         cell_in_roi = 0
#         for j in range(len(marker_x)): 
#             code = Point(marker_x[j], marker_y[j])
#             if code.within(polygon):
#                 cell_in_roi += 1
                
#         mslabel = None
#         if msGroup[image_no] == 'mptp':
#             mssave3[1].append([image_no, cell_in_roi])
#             mslabel = 'MPTP'
#         elif  msGroup[image_no] == 'saline':
#             mssave3[0].append([image_no, cell_in_roi])
#             mslabel = 'saline'
            
#         mssave4.append([image_no, cell_in_roi, mslabel])

# Aprism = pd.DataFrame(np.array(mssave3[0])[:,-1])
# Aprism = pd.concat((Aprism, pd.DataFrame(np.array(mssave3[1])[:,-1])), axis=1, ignore_index=True)

# Aprism2 = pd.DataFrame(np.array(mssave4))




















































