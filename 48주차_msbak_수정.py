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
from PIL import Image,ImageEnhance  

#%% """HYPERPARAMETERS"""

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
                 length=None, width=None):
    
    import numpy as np
    square = im[find_min(y, unit):find_max(y, unit, length), find_min(x,unit):find_max(x, unit,width)]
    extend_x = size - find_max(x, unit,width) + find_min(x,unit)
    extend_y = size - find_max(y, unit,length) + find_min(y,unit)
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
    from PIL import Image,ImageFilter, ImageEnhance, ImageOps
    
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

#%% keras setup
def model_setup(xs=None, ys=None, lr=1e-6):
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models, regularizers
    from tensorflow.keras.layers import BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    model = models.Sequential()
    model.add(layers.Conv2D(2**6, (2, 2), activation='relu', input_shape=xs))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**6, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(2**6, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(2**6, activation='relu' ))
    model.add(layers.Dense(2**6, activation='sigmoid') )
    model.add(layers.Dense(ys, activation='softmax'))        

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               # loss='binary_crossentropy',

    #               metrics=['accuracy'])

    model.compile(optimizer=Adam(learning_rate=lr, decay=1e-6, beta_1=0.9, beta_2=0.999), \
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = model_setup(xs=(29,29,3), ys=(2))
print(model.summary())
#%% XYZgen """train X, Y 만들기"""

ms_filepath = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'data_48.pickle'
with open(ms_filepath, 'rb') as file:
    dictionary = pickle.load(file)

ms_filepath2 = 'C:\\SynologyDrive\\study\\dy\\48\\' + 'roipoints_48.pickle'
with open(ms_filepath2, 'rb') as file:
    dictionary2 = pickle.load(file)

keylist = list(dictionary.keys())
    
prev_len=0
total_len = 0

X, Y, Z = [], [], []

for n_num in tqdm(range(len(keylist))):
    n = keylist[n_num]
    
    if not n in [31, 'A3.1L']:

        marker_x = dictionary[n]['marker_x']
        marker_y = dictionary[n]['marker_y']
        width = dictionary[n]['width']
        length = dictionary[n]['length']
        im = dictionary[n]['imread']
        
        int_value = []
        # positive crop
        for i in range(len(marker_x)):
            crop = find_square(y=marker_y[i], x=marker_x[i], unit=size_hf, im=im, length=length, width=width)
            
            X.append(crop)
            Y.append([0,1])
            Z.append([n, marker_y[i], marker_x[i]])

            int_value.append(im_mean(crop))
            
        int_thr = int(round(np.mean(int_value)))
        
        # negative crop
        negative_cnt = 0
        occupied = np.transpose(np.array([marker_x, marker_y]))
        
        epochs = 10000000
        for epoch in range(epochs):
            if negative_cnt > len(marker_x) * 3: break

            A = random.randrange(min(marker_y)+1+size, max(marker_y)-1-size)
            B = random.randrange(min(marker_x)+1+size, max(marker_x)-1-size)
            
            passsw = True
            
            # condition 1
            for yy in range(len(occupied)):
                if (occupied[yy][0] - (2*sh)< A < occupied[yy][0] + (2*sh) ) and (occupied[yy][1] - (2*sh) < B < occupied[yy][0] + (2*sh)):
                    passsw = False; break

            pc = find_square(y=A, x=B, \
                             unit=size_hf, im=im, length=length, width=width)
            neg_int = im_mean(pc)
            
            # condition 2
            if neg_int > int_thr + nct: passsw = False
        
            if passsw:
                occupied = np.concatenate((occupied, np.array([[A, B]])), axis=0)
                negative_cnt += 1
                
                X.append(pc)
                Y.append([1,0])
                Z.append([n, marker_y[i], marker_x[i]])
            if epochs == epoch: print(n, 'ng shorts')

#%% cvset
X = np.array(X); Y = np.array(Y); Z = np.array(Z)
print(X.shape, len(X), 'np.sum(Y, axis=0)', np.sum(Y, axis=0))

# LOSO cv
tlist = list(range(len(X)))
cvlist = []
# print(Z_vix[:,3])
session_list = list(set(np.array(Z[:,0])))

for se in range(len(session_list)):
    msid = session_list[se]
    telist = np.where(np.array(Z[:,0])==msid)[0]
    # print(se, msid, len(cvlist))     
    cvlist.append([telist, msid])

print('len(cvlist)', len(cvlist))

#%% cv training
cv = 0
for cv in range(len(cvlist)):
    telist = cvlist[cv][0]
    trlist = list(set(tlist)-set(telist))
    
    X_tr = X[trlist]; Y_tr = Y[trlist]; Z_tr = Z[trlist]
    X_te = X[telist]; Y_te = Y [telist]; Z_te = Z[telist]
    
    print(len(X_tr), len(X_te))
    print(np.mean(Y_tr, axis=0), np.mean(Y_te, axis=0))
    
    X_aug, Y_aug, Z_aug = [], [], []
    for i in tqdm(range(len(X_tr))):
        xout, yout, zout = im_aug(X_tr[i], Y_tr[i], Z_tr[i])
        X_aug += list(xout)
        Y_aug += list(yout)
        Z_aug += list(zout)
    X_aug, Y_aug, Z_aug = np.array(X_aug), np.array(Y_aug), np.array(Z_aug)
        
    X_tr = np.concatenate((X_tr, X_aug), axis=0)
    Y_tr = np.concatenate((Y_tr, Y_aug), axis=0)
    Z_tr = np.concatenate((Z_tr, Z_aug), axis=0)
    
    print(X_tr.shape)
    print(np.mean(Y_tr, axis=0), np.mean(Y_te, axis=0))

    #%%

    model = model_setup(xs=X_tr[0].shape, ys=Y_tr[0].shape[0])
    model.fit(X_tr, Y_tr, epochs=1000, verbose=1, validation_data=(X_te, Y_te))
    

    #%% test all set
    test_image_no = cvlist[cv][1]
    width = dictionary[test_image_no]['width']
    height = dictionary[test_image_no]['length']
    t_im = dictionary[test_image_no]['imread']
    # marker_x = dictionary[test_image_no]['marker_x']
    # marker_y = dictionary[test_image_no]['marker_y']
    # occupied = np.transpose(np.array([marker_x, marker_y]))
    
    if False:
        polygon = dictionary2[test_image_no]['polygon']
        points = dictionary2[test_image_no]['points']
        
        # note, 133 없음?

    X_total_te, Z_total_te = [], []
    for row in tqdm(range(sh, t_im.shape[0]-sh)):
        for col in range(sh, t_im.shape[1]-sh):
            crop = find_square(y=row, x=col, unit=sh, im=t_im, length=length, width=width)
            
            # note, padding이 양쪽으로 되는거 같은데
            
            X_total_te.append(crop)
            Z_total_te.append([row, col])
            
    X_total_te = np.array(X_total_te)
    Z_total_te = np.array(Z_total_te)
    
    yhat = model.predict(X_total_te)
    
    allo = np.zeros((t_im.shape[0], t_im.shape[1]))
    for i in range(len(yhat)):
        row = Z_total_te[i][0]
        col = Z_total_te[i][1]
        allo[row, col] = yhat[i,1]

    # if False:
    #     plt.imshow(allo > 0.9)
    #     positive_pred = []
    #     for i in range(len(occupied)):
    #         positive_pred.append(allo[occupied[i,1], occupied[i,0]])
    #     print('np.mean(positive_pred)', np.mean(positive_pred))

    #%% >>> allo
    
    threshold = 0.5
    contour_thr = 40
    
    def get_F1(threshold=None, contour_thr=None,\
               height=height, width=width, yhat=yhat, Z_total_te=Z_total_te, t_im=t_im):
    
        import numpy as np
        import cv2
        
        noback_img = np.zeros((height, width, 3), np.uint8)
        noback_img[:]= [255,255,0]
        
        vix = np.where(yhat[:,1] > threshold)[0]
        for i in vix:
            row = Z_total_te[i][0]
            col = Z_total_te[i][1]
            noback_img[row,col] = [0,0,255]
        
        
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
                # code = Point(cx,cy)
                # if code.within(polygon)==True:
                pink.append((cx,cy))
                los.append(cnt) 
        
        # predicted area 
        white = np.zeros((height,width,3))*np.nan
        white[:,:] = [255,255,255]
        for lo in los:
            cv2.drawContours(white, [lo], 0, (0, 0, 0), -2)  # black
        
        colored = []
        w = np.where(white[:,:,0]==0)
        for i in range(len(w[0])):
            colored.append((w[1][i],w[0][i]))
        
        # answer (positive) area
        co = []
        for i in vix:
            row = Z_total_te[i][0]
            col = Z_total_te[i][1]
            # code = Point(w,h)
            # if code.within(polygon)==True:
            co.append((int(col),int(row)))
            
        original_img = np.array(t_im, dtype=np.uint8)
        for nn in range(len(co)):
            cv2.circle(original_img, co[nn], 3, (0,255,255), -1)  
            
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
        tparea = []
        w = np.where(white2[:,:,0]==0)
        for i in range(len(w[0])):
            tparea.append((w[1][i],w[0][i]))     
    
        # predicted cells area
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
        # score = 2*precision*recall / (precision+recall)
        TP = tp
        FP = fp
        FN = fn
        precision = tp/(tp+fp)
        recall = tp / (tp+fn)
        F1_score = 2 * precision * recall / (precision + recall)
        
        return F1_score

    mssave = []
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        for contour_thr in [10, 20, 40, 50]:
            F1_score = get_F1(threshold=threshold, contour_thr=contour_thr,\
                       height=height, width=width, yhat=yhat, Z_total_te=Z_total_te, t_im=t_im)
                
            mssave.append([threshold, contour_thr, F1_score])
            print([threshold, contour_thr, F1_score])































































