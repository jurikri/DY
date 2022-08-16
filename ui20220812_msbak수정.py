# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:15:50 2022

@author: SKKLab
새로운 모델 메모리 이슈 해결한 상태로 적용
"""
## waitkey 때문에 종료안되는 문제 해결 (quit 감)
# /3 ver week 20

from PyQt5 import QtCore, QtGui, QtWidgets

import sys, cv2, imutils, time, csv, keyboard
# pip install imutils
# pip install keyboard

import cv2 as cv
import numpy as np
import webbrowser

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip,  QFrame, QColorDialog, QFileDialog
                             , QRadioButton, QDesktopWidget, QCheckBox, QMenu, QGridLayout, QVBoxLayout
                             , QLabel, QComboBox)
from PyQt5.QtGui import QFont, QColor, QIcon, QImage, QPainter, QPen
from PyQt5.QtCore import Qt

#%%

class Ui_MainWindow(object):
#% Default settings
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(957, 168)
        MainWindow.move(0,0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_next1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_next1.setGeometry(QtCore.QRect(287, 38, 21, 21))
        self.pushButton_next1.setObjectName("pushButton_next1")
        self.groupBox_BC = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_BC.setGeometry(QtCore.QRect(15, 10, 191, 71))
        self.groupBox_BC.setObjectName("groupBox_BC")
        self.label_valueC = QtWidgets.QLabel(self.groupBox_BC)
        self.label_valueC.setGeometry(QtCore.QRect(160, 44, 21, 16))
        self.label_valueC.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valueC.setObjectName("label_valueC")
        self.horizontalSlider_C = QtWidgets.QSlider(self.groupBox_BC)
        self.horizontalSlider_C.setGeometry(QtCore.QRect(32, 44, 121, 16))
        self.horizontalSlider_C.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_C.setObjectName("horizontalSlider_C")
        self.horizontalSlider_B = QtWidgets.QSlider(self.groupBox_BC)
        self.horizontalSlider_B.setGeometry(QtCore.QRect(32, 21, 121, 16))
        self.horizontalSlider_B.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_B.setObjectName("horizontalSlider_B")
        self.label_C = QtWidgets.QLabel(self.groupBox_BC)
        self.label_C.setGeometry(QtCore.QRect(11, 44, 16, 16))
        self.label_C.setAlignment(QtCore.Qt.AlignCenter)
        self.label_C.setObjectName("label_C")
        self.label_valueB = QtWidgets.QLabel(self.groupBox_BC)
        self.label_valueB.setGeometry(QtCore.QRect(160, 21, 21, 16))
        self.label_valueB.setAlignment(QtCore.Qt.AlignCenter)
        self.label_valueB.setObjectName("label_valueB")
        self.label_B = QtWidgets.QLabel(self.groupBox_BC)
        self.label_B.setGeometry(QtCore.QRect(11, 21, 16, 16))
        self.label_B.setAlignment(QtCore.Qt.AlignCenter)
        self.label_B.setObjectName("label_B")
        self.groupBox_crop = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_crop.setGeometry(QtCore.QRect(215, 10, 61, 71))
        self.groupBox_crop.setObjectName("groupBox_crop")
        self.pushButton_crop = QtWidgets.QPushButton(self.groupBox_crop)
        self.pushButton_crop.setGeometry(QtCore.QRect(10, 19, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Algerian")
        font.setPointSize(11)
        self.pushButton_crop.setFont(font)
        self.pushButton_crop.setObjectName("pushButton_crop")
        self.groupBox_prediction = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_prediction.setGeometry(QtCore.QRect(317, 10, 231, 71))
        self.groupBox_prediction.setObjectName("groupBox_prediction")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_prediction)
        self.doubleSpinBox.setGeometry(QtCore.QRect(13, 37, 51, 22))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox.setValue(0.5)
        self.pushButton_color1 = QtWidgets.QPushButton(self.groupBox_prediction)
        self.doubleSpinBox.setRange(0.01,0.99)
        self.doubleSpinBox.setSingleStep(0.01)
        self.pushButton_color1.setGeometry(QtCore.QRect(71, 19, 41, 41))
        self.pushButton_color1.setObjectName("pushButton_color1")
        self.label_level = QtWidgets.QLabel(self.groupBox_prediction)
        self.label_level.setGeometry(QtCore.QRect(19, 18, 31, 16))
        self.label_level.setObjectName("label_level")
        self.pushButton_select = QtWidgets.QPushButton(self.groupBox_prediction)
        self.pushButton_select.setGeometry(QtCore.QRect(125, 19, 41, 41))
        self.pushButton_select.setObjectName("pushButton_select")
        self.pushButton_reset1 = QtWidgets.QPushButton(self.groupBox_prediction)
        self.pushButton_reset1.setGeometry(QtCore.QRect(179, 19, 41, 41))
        self.pushButton_reset1.setObjectName("pushButton_reset1")
        self.groupBox_celldetection = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_celldetection.setGeometry(QtCore.QRect(595, 10, 201, 71))
        self.groupBox_celldetection.setObjectName("groupBox_celldetection")
        self.pushButton_color2 = QtWidgets.QPushButton(self.groupBox_celldetection)
        self.pushButton_color2.setGeometry(QtCore.QRect(71, 19, 41, 41))
        self.pushButton_color2.setObjectName("pushButton_color2")
        self.label_size = QtWidgets.QLabel(self.groupBox_celldetection)
        self.label_size.setGeometry(QtCore.QRect(22, 18, 31, 16))
        self.label_size.setObjectName("label_size")
        self.doubleSpinBox2 = QtWidgets.QDoubleSpinBox(self.groupBox_celldetection)
        self.doubleSpinBox2.setGeometry(QtCore.QRect(11, 37, 51, 22))
        self.doubleSpinBox2.setObjectName("doubleSpinBox2")
        self.pushButton_next3 = QtWidgets.QPushButton(self.groupBox_celldetection)
        self.pushButton_next3.setGeometry(QtCore.QRect(121, 30, 21, 21))
        self.pushButton_next3.setObjectName("pushButton_next3")
        self.pushButton_reset2 = QtWidgets.QPushButton(self.groupBox_celldetection)
        self.pushButton_reset2.setGeometry(QtCore.QRect(152, 19, 41, 41))
        self.pushButton_reset2.setObjectName("pushButton_reset2")
        self.pushButton_next2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_next2.setGeometry(QtCore.QRect(560, 38, 21, 21))
        self.pushButton_next2.setObjectName("pushButton_next2")
        self.groupBox_imshow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_imshow.setGeometry(QtCore.QRect(811, 10, 131, 71))
        self.groupBox_imshow.setObjectName("groupBox_imshow")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_imshow)
        self.radioButton.setGeometry(QtCore.QRect(10, 18, 101, 16))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_imshow)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 35, 121, 16))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_imshow)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 52, 121, 16))
        self.radioButton_3.setObjectName("radioButton_3")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(826, 99, 118, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_status2 = QtWidgets.QLabel(self.centralwidget)
        self.label_status2.setGeometry(QtCore.QRect(554, 101, 261, 20))
        self.label_status2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_status2.setObjectName("label_status2")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(15, 82, 921, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_counts = QtWidgets.QLabel(self.centralwidget)
        self.label_counts.setGeometry(QtCore.QRect(18, 101, 51, 16))
        self.label_counts.setObjectName("label_counts")
        self.label_valuecounts = QtWidgets.QLabel(self.centralwidget)
        self.label_valuecounts.setGeometry(QtCore.QRect(73, 101, 31, 16))
        self.label_valuecounts.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_valuecounts.setObjectName("label_valuecounts")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 957, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSave = QtWidgets.QMenu(self.menuFile)
        self.menuSave.setObjectName("menuSave")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionImage = QtWidgets.QAction(MainWindow)
        self.actionImage.setObjectName("actionImage")
        self.actionMarkers = QtWidgets.QAction(MainWindow)
        self.actionMarkers.setObjectName("actionMarkers")
        self.actionUser_Guide = QtWidgets.QAction(MainWindow)
        self.actionUser_Guide.setObjectName("actionUser_Guide")
        self.menuSave.addAction(self.actionImage)
        self.menuSave.addAction(self.actionMarkers)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.menuSave.menuAction())
        self.menuHelp.addAction(self.actionUser_Guide)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)

        self.groupBox_crop.setEnabled(False)
        self.groupBox_prediction.setEnabled(False)
        self.pushButton_next2.setEnabled(False)
        self.groupBox_imshow.setEnabled(False)
        self.doubleSpinBox2.setEnabled(False)
        self.pushButton_color2.setEnabled(False)
        self.pushButton_next3.setEnabled(False)
        self.pushButton_reset2.setEnabled(False)
        self.groupBox_BC.setEnabled(False)
        self.pushButton_next1.setEnabled(False)
        self.groupBox_celldetection.setEnabled(False)
        
        
        
        
        
        
        
#%% variables
        self.width = 0
        self.height = 0
        self.test_Z = None


        self.image = None # loaded image
        self.nocrop = self.image # Brightness와 Contrast: adjusted // cropped: No  // size changed?: no
        self.crop = self.image # B and C: No // cropped: No // size: no
        self.square = self.image # latest ver. of self.crop (prevent an error when pressing esc while selecting a rect roi)
        self.nosize = self.image # B and C: yes // cropped: yes // size : no
        self.h = None
        self.tmp = self.image
        self.tmp2 = self.tmp
        self.cellcounting = None
        
        self.contrast_value_now = 1 # Updated contrast value
        self.brightness_value_now = 0 # updated brightness value

        self.cropped_im = None
        self.col = [255,255,0]
        self.col2 = [255,255,0]
        self.col_img = None
        self.com_img = None

        self.prediction_list = None
        self.result_list = None
        self.cropped_im = None
        self.prediction_mask = None
        self.background_mask = None
        
        self.original_pred_mask = None
        self.original_back_mask = None    
        self.finalprediction = None
        self.center = []
        self.cellcounting = None      
        self.original_center = []
        
        self.status_now = ''
        
        self.winx = 0
        self.winy = 200
        
        self.dots=[]
        self.pred_thr = 0.5
        self.pred = None
        self.stop_waiting = 0

        ##
        self.win_size = 0
        
#%% triggers 

        self.horizontalSlider_B.valueChanged['int'].connect(self.label_valueB.setNum)
        self.horizontalSlider_C.valueChanged['int'].connect(self.label_valueC.setNum)
        self.horizontalSlider_B.valueChanged['int'].connect(self.brightness_value)
        self.horizontalSlider_C.valueChanged['int'].connect(self.contrast_value)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # MENU
        self.actionOpen.triggered.connect(self.loadImage) # 이미지 열기
        # self.actionOpen.triggered.connect(self.adjust) love
        self.actionImage.triggered.connect(self.savePhoto) # 이미지 저장
        self.actionMarkers.triggered.connect(self.save_markers)
        self.actionUser_Guide.triggered.connect(self.user_guide)
        # pushButtons
        self.pushButton_crop.clicked.connect(self.drawRect)
        self.pushButton_next1.clicked.connect(self.prediction)
        self.pushButton_next1.clicked.connect(self.predstart)

        self.pushButton_color1.clicked.connect(self.showDialog)
        self.pushButton_select.clicked.connect(self.drawRects)
        self.pushButton_reset1.clicked.connect(self.reset)
        self.pushButton_next2.clicked.connect(self.settingstep3)
        self.pushButton_next2.clicked.connect(self.settingbutton3)
        self.doubleSpinBox.valueChanged.connect(self.settingstep2)
        self.doubleSpinBox2.valueChanged.connect(self.settingstep3)
        self.pushButton_color2.clicked.connect(self.showDialog2)
        self.pushButton_next3.clicked.connect(self.settingstep4)
        self.pushButton_next3.clicked.connect(self.points)
        self.pushButton_next3.clicked.connect(self.clickstart)

        self.pushButton_reset2.clicked.connect(self.reset2)
        
        # radioButtons
        self.radioButton_2.clicked.connect(lambda: self.setPhoto2(self.prediction_list))
        self.radioButton.clicked.connect(lambda: self.setPhoto2(self.cropped_im))
        self.radioButton_3.clicked.connect(lambda: self.setPhoto2(self.result_list))
        
        
        
        

#%% functions
    #%% adjust
    def adjust(self):
        self.label_status2.setText('Processing ... Please wait')
    def user_guide(self):   
        webbrowser.open_new('https://neuroglia.khu.ac.kr/home')
        
    def get_contrast(self, img):
        import numpy as np

        from scipy.stats import mode
        import numpy as np
        pc = np.array(img)
        pc = (pc).astype(np.uint8)
        pc =cv2.cvtColor(pc, cv2.COLOR_BGR2RGB)
        pc=cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY)
        std = np.std(pc)
        c=  min(mode(pc)[0][0])
        u = pc - c
        u = u**2
        a = np.sum(u)
        a = a / (len(pc)*len(pc[0]))
        a = np.sqrt(a)
        
    
        b = np.mean(pc)
        c = min(mode(pc)[0][0])
        return a, b ,c,std
    ####
    def image_enh(self,im):
        from PIL import Image,ImageEnhance
        im = Image.open(im)
        self.progressBar.setValue(20)
        QtWidgets.QApplication.processEvents()
        self.label_status2.setText('Processing ... Please wait')
        self.label_status2.repaint()
        print('(adjusting brightness and constrast automatically...)')
        nn = []
        t = []
        s = []
        l = []
        #brightness (mean)
        oo = 0
        for i in range(0,50):
        #Display image  
            en = ImageEnhance.Brightness(im).enhance(0.1+oo)
            if (253>self.get_contrast(en)[1]>241):
                break
            else:
                oo += 0.1   
        self.progressBar.setValue(60)
        QtWidgets.QApplication.processEvents()
        # contrast (sig)
        ii = 0
        for i in range(50):
            
            enh = ImageEnhance.Contrast(en).enhance(0.2+ii)
            if 36<self.get_contrast(enh)[3]<43:
                break
            elif self.get_contrast(enh)[3]>43:
                break
            
            else:
                ii += 0.1
        print('done')
        self.progressBar.setValue(100)
        QtWidgets.QApplication.processEvents()
        time.sleep(1)
        self.progressBar.setValue(0)
        ad_img = np.array(enh)    
        e = cv2.cvtColor(ad_img,cv2.COLOR_BGR2RGB)
        return e
    #%% fx - open image
    
    def loadImage(self): # image open
        ####
        from PIL import Image
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        # self.image = cv2.imread(self.filename)
        self.image = self.image_enh(self.filename)
        self.nocrop = self.image
        self.square = self.image
        self.label_status2.setText('')
        # enable and disable
        
        # enable
        self.groupBox_BC.setEnabled(True)   
        self.groupBox_crop.setEnabled(True)
        self.pushButton_next1.setEnabled(True)
        # disable

        self.groupBox_prediction.setEnabled(False)
        self.groupBox_celldetection.setEnabled(False)
        self.pushButton_next2.setEnabled(False)
        self.groupBox_imshow.setEnabled(False)
        self.label_valueB.setText('0')
        self.label_valueC.setText('1')
        self.setPhoto(self.image)
        self.pushButton_reset2.setEnabled(False)
        
        self.label_valuecounts.setText('0')
        self.doubleSpinBox2.setValue(0.0)
        
        self.square = self.image
        self.nocrop = self.image
        self.crop = self.image 
        self.tmp = self.image
        self.tmp2 = self.tmp
        self.nosize = self.image # B and C: yes // cropped: yes // size : no
        self.h = None     
        self.cellcounting = None
        
        self.contrast_value_now = 1 # Updated contrast value
        self.brightness_value_now = 0 # updated brightness value
        self.horizontalSlider_B.setValue(0)
        self.horizontalSlider_C.setValue(1)
        self.cropped_im = None
        self.col = [255,255,0]
        self.col2 = [255,255,0]
        self.col_img = None
        self.com_img = None

        self.prediction_list = None
        self.result_list = None
        self.prediction_mask = None
        self.background_mask = None
        
        self.original_pred_mask = None
        self.original_back_mask = None     
        self.finalprediction = None
        self.center = []
        self.cellcounting = None      
        self.original_center = []   
        self.switch()

        self.dots=[]
        self.pred_thr = 0.5
        self.pred = None
        self.stop_waiting = 0

        self.win_size = 0
        #
        
    #%% fx - setPhoto
    def setPhoto(self, img):
        self.tmp = img
        self.h = img.shape[0]*2**self.win_size
        img = imutils.resize(img, height=int(self.h))
        cv2.imshow('', img)
        cv2.moveWindow('', self.winx, self.winy)
        
    def setPhoto2(self, img):
        self.tmp2 = img
        self.h = img.shape[0]*2**self.win_size
        img = imutils.resize(img, height=int(self.h))
        cv2.imshow('', img)
        cv2.moveWindow('', self.winx, self.winy)

    def setPhoto3(self):
        img = self.nosize.copy()
        self.h = self.nosize.shape[0]*2**self.win_size
        img = imutils.resize(img, height=int(self.h))
        for i in range(len(self.center)):
            cv.circle(img, self.mul_centers(2**(self.win_size),self.center)[i], int(2*(2**self.win_size)), self.col2, -1) 
        cv2.imshow('', img)
        cv2.moveWindow('', self.winx, self.winy) 
        
        
    #%% fx - crop      
    def drawRect(self):
        cv2.destroyAllWindows()
        img = self.nocrop
        self.h = img.shape[0]*2**self.win_size
        img = imutils.resize(img, height = int(self.h))
   
        x,y,w,h	= cv2.selectROI('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', img, False)
        cv2.moveWindow('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', self.winx, self.winy)
        if w and h:
            roi = img[y:y+h, x:x+w]
        
            mul = 2** (self.win_size *-1)
            self.square = self.image[int(y*mul):int((y+h)*mul),int(x*mul):int((x+w)*mul)]
            self.crop = self.image[int(y*mul):int((y+h)*mul),int(x*mul):int((x+w)*mul)]
            self.nosize = self.nocrop[int(y*mul):int((y+h)*mul),int(x*mul):int((x+w)*mul)]

        cv2.destroyAllWindows()

        # if img_width > 1000:
        #     img = imutils.resize(img, width = int(img_width/2))
        #     x,y,w,h	= cv2.selectROI('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', img, False)
        #     cv2.moveWindow('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', self.winx, self.winy)
            
        #     if w and h:
        #         roi = img[y:(y+h), x:(x+w)]
        #     self.square = self.image[y*2:(y+h)*2, x*2:(x+w)*2]
        #     self.crop = self.image[y*2:(y+h)*2, x*2:(x+w)*2]
        #     cv2.destroyAllWindows()
        #     self.setPhoto2(roi)
            
        # else:
        #     x,y,w,h	= cv2.selectROI('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', img, False)
        #     cv2.moveWindow('DRAG to draw a rectangle. After selecting the area, PRESS ENTER to nail it down. PRESS ESC to quit without selecting.', self.winx, self.winy)
        #     if w and h:
        #         roi = img[y:y+h, x:x+w]
        #     self.square = self.image[y:y+h,x:x+w]
        #     self.crop = self.image[y:y+h,x:x+w]
        #     cv2.destroyAllWindows()

        self.setPhoto2(self.nosize)
        
    #%% fx - brightness and contrast

# brightness


    def brightness_value(self,value): # take value form the slider
        self.brightness_value_now = value
        self.update()
        self.update_no_crop()

    def changeBrightness(self, img, value): # take img and brightness value & perform brightness change 
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        lim = 255 -value
        v[v>lim] = 255
        v[v<lim]+= value
        final_hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
# contrast

            
    def contrast_value(self,value):
      self.contrast_value_now = value
      self.update()
      self.update_no_crop()
        
    def changecontrast(self,img,value):
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab) 
      clahe = cv2.createCLAHE(clipLimit=value/20, tileGridSize=(8,8))
      cl = clahe.apply(l)
      limg = cv2.merge((cl, a, b))
      img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
      return img


    #%% fx - update functions (update Crop, brightness, contrast value)
    # update the photo according to the current values and set it to photo label.
    def update(self):
        if len(self.crop)==0:
            self.crop = self.square
            
        img = self.changeBrightness(self.crop, self.brightness_value_now)
        img = self.changecontrast(img,self.contrast_value_now)
        self.nosize = img
        self.tmp2 = img
        self.setPhoto(img)
        
    def update_no_crop(self):
        img = self.changeBrightness(self.image, self.brightness_value_now)
        img = self.changecontrast(img,self.contrast_value_now)
        self.nocrop = img

    #%% fx - freeze while prediction
    def predstart(self):    
        self.stop_waiting=1
        
    def clickstart(self):
        self.stop_waiting = 3
        self.lastmodi()


    #%% fx - model prediction 

    
    def prediction(self):
        self.groupBox_BC.setEnabled(False)
        self.groupBox_crop.setEnabled(False)
        self.pushButton_next1.setEnabled(False)
        self.label_status2.setText('Detecting ... Please Wait')
        self.status_now = 'cell detection'
        self.stop_waiting = 1
        self.label_status2.repaint()


        
        #crop 크기 수정------------------                                    
        size_hf = 14 # crop size: 29 x 29 
        

        #########변수이름변경###########
        sh = size_hf
        size = (size_hf*2)+1
        ###############################
        ### keras setup #############################################################
        print('\n(setting up model...)')
        xs = (size,size,3)
        ys = 2 
        self.progressBar.setValue(10)
        QtWidgets.QApplication.processEvents()
        
        def model_setup(xs=None, ys=None, lr=1e-4):
            import tensorflow as tf
            from tensorflow.keras import datasets, layers, models, regularizers
            from tensorflow.keras.layers import BatchNormalization, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = models.Sequential()
            model.add(layers.Conv2D(2**8, (4, 4), activation='relu', input_shape=xs))
            model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Conv2D(2**8, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Conv2D(2**8, (2, 2), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Flatten())
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='relu' ))
            model.add(layers.Dense(2**7, activation='sigmoid') )
            model.add(layers.Dense(ys, activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=lr, decay=1e-3, beta_1=0.9, beta_2=0.999), \
                          loss='categorical_crossentropy', metrics=['accuracy']) 
            
            return model
        
        model = model_setup(xs=(29,29,3), ys=(2))
        
        weight_path = 'C:\\SynologyDrive\\study\\dy\\52\\weightsave_finalcheck\\cv_0_subject_A3_1L_total_final.h5'
        model.load_weights(weight_path)
        
        ### 5th cell 총 크롭 수 계산#############################################
        """test image 총 크롭 수를 계산"""
        

        self.cropped_im = self.nosize
        self.col_img = np.zeros((self.cropped_im.shape[0], self.cropped_im.shape[1], 3 )) * np.nan
        self.com_img = np.zeros((self.cropped_im.shape[0], self.cropped_im.shape[1], 3)) * np.nan
        ####################################################################
        self.progressBar.setValue(20)
        QtWidgets.QApplication.processEvents()
        print("\n(5th cell - calculating the number of total crops...)")
        t_crops = (self.cropped_im.shape[1]-size+1) * (self.cropped_im.shape[0]-size+1) # padding된 roi 속 모든 crop 갯수


        ### 6th cell roi 크롭########################################################
        """image 원본으로 size_hf씩 확장이 불가능할 때 white로 padding"""
        print("\n(6th cell - padding ...)")
        
        # roi cropped image를 padding (상하좌우 1 pixel씩)
        p_roi_im = self.cropped_im
        p_roi_len = p_roi_im.shape[0] #세로길이
        p_roi_wid = p_roi_im.shape[1] #가로길이
        
        print('padded roi image length:', p_roi_len)
        print('padded roi image width:', p_roi_wid)
        
        q = int((p_roi_len+1-size)/3)
        r = p_roi_len+1-size - (int((p_roi_len+1-size)/3) *3)
        print('roi image 속 총 crop 수:', t_crops)
        size = (sh * 2) + 1
        num=0
        lenlist = []
        for i in range(2):
            lenlist.append(q)
        lenlist.append(q+r)
        
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
        
        
        def ms_prep_and_predict(t_im=None, sh=14, \
                                find_square=find_square, model=None, 
                                divnum=10):
            
            """
            t_im = test할 image
            sh = half size (14로 고정)
            rowmin=None, rowmax=None, colmin=None, colmax=None
            ->  image에 모든 영역을 test 하지않고, 일부분만 test하기 위한 변수들
                사용자가 직접 지정하거나, 사용자가 지정한 ROI를 기반으로 결정하면 될듯
            
            find_square = 사용자 정의함수 그대로 받음
            model = keras 모델, 학습된 weight 까지 load 한다음 전달 할것
            
            polygon = test 할 image의 polygon
            (polygon으로 rowmin, rowmax, colmin, olmax를 정하게 해도 될듯)
            
            divnum = memory 부족 문제 해결을 위해 몇번에 걸쳐 나눌건지
            
            """
            from shapely.geometry import Point
            import numpy as np
            
            height = t_im.shape[0]
            width = t_im.shape[1]
        
            # allo = np.zeros((t_im.shape[0], t_im.shape[1])) * np.nan
            yhat_save = []
            z_save = []
            
            forlist = list(range(height))
            div = int(len(forlist)/divnum)
            for div_i in range(divnum):
                print('div', div_i)
                if div_i != divnum-1: forlist_div = forlist[div_i*div : (div_i+1)*div]
                elif div_i== divnum-1: forlist_div = forlist[div_i*div :]
            
                for row in forlist_div:
                    X_total_te = []
                    Z_total_te = []
                    for col in range(width):
                        y=row; x=col; unit=sh; im=t_im; height=height; width=width
                        crop = find_square(y=y, x=x, unit=unit, im=im, height=height, width=width)
                        if crop.shape == (29, 29, 3):
                            
                            # note, 크기가 안맞는 경우가 있음
                        # note, padding이 양쪽으로 되는거 같은데
                            # crop = crop / np.mean(crop)
                            # if not(polygon is None):
                            #     code = Point(col,row)
                            #     if code.within(polygon):
                            #         X_total_te.append(crop)
                            #         Z_total_te.append([row, col])
                            # else:
                            X_total_te.append(crop)
                            Z_total_te.append([row, col])
                    
                    if len(X_total_te) > 0:
                        X_total_te = np.array(X_total_te)
                        yhat = model.predict(X_total_te, verbose=0)
                        for i in range(len(yhat)):
                            row = Z_total_te[i][0]
                            col = Z_total_te[i][1]
                            # allo[row, col] = yhat[i,1]
                        yhat_save += list(yhat)
                        z_save += Z_total_te
                        
            z_save = np.array(z_save)
            msdict = {'yhat_save': yhat_save, 'z_save': z_save}
            return msdict
        
        t_im = self.cropped_im
        model = model
        
        msdict = ms_prep_and_predict(t_im=t_im, sh=14, \
                                find_square=find_square, model=model, 
                                divnum=10)
            
        yhat_save = msdict['yhat_save']
        z_save = msdict['z_save']
        
        ix = np.array(list(range(len(z_save))))
        ix = np.reshape(ix, (len(ix), 1))
        z_save2 = np.concatenate((ix, z_save), axis=1)
        
        self.pred = np.array(yhat_save)
        self.test_Z = z_save2
        
        self.progressBar.setValue(30)
        QtWidgets.QApplication.processEvents()
        self.label_status2.setText('Detecting ... Please wait')
        self.label_status2.repaint()

        
        self.width = p_roi_im.shape[1]
        self.height = p_roi_im.shape[0]
        print('\n\n (consisting predicted image...)')
        
        background_img = np.zeros((self.height, self.width,3),) * np.nan
        prediction_img = np.zeros((self.height, self.width,3),) * np.nan
        background_img[:] = [255, 255, 255] 
        prediction_img[:]= [0,0,0]
        
        a,b,c, = self.col
        self.col_img[:] = [self.col]
        self.com_img[:] = [255-a, 255-b, 255-c]
        self.col_img = self.col_img.astype('uint8')
        self.com_img = self.com_img.astype('uint8')
        self.progressBar.setValue(90)
        QtWidgets.QApplication.processEvents()
        
        background_img[:] = [255, 255, 255] 
        prediction_img[:]= [0,0,0]
        vix = np.where(self.pred[:,1] > self.pred_thr)[0]
        for i in range(len(self.test_Z)):
    
            if i in vix:
                w = int(self.test_Z[i,2])
                h = int(self.test_Z[i,1])
                background_img[h,w] = [0,0,0]
                prediction_img[h,w] = [255,255,255]
        prediction_img = prediction_img.astype('uint8')
        background_img = background_img.astype('uint8')
        self.prediction_mask = prediction_img
        self.background_mask = background_img
        self.original_pred_mask=prediction_img
        self.original_back_mask=background_img

        self.progressBar.setValue(100)
        QtWidgets.QApplication.processEvents()
        time.sleep(1)
        self.progressBar.setValue(0)
        QtWidgets.QApplication.processEvents()        
        self.radioButton_3.setChecked(True)
        self.make_prediction()
        self.make_result()
        self.setResult()
        
        print('done')
        self.label_status2.setText(self.status_now)
        self.label_status2.repaint()
       
        
        self.groupBox_prediction.setEnabled(True)
        self.groupBox_imshow.setEnabled(True)
        self.pushButton_next2.setEnabled(True)
        self.mode()
    
    #%% fx - make prediction and result     
    def settingstep2(self):
        background_img = np.zeros((self.height, self.width,3)) * np.nan
        prediction_img = np.zeros((self.height, self.width,3)) * np.nan
        background_img[:] = [255, 255, 255] 
        prediction_img[:]= [0,0,0]
        
        a,b,c, = self.col
        self.col_img[:] = [self.col]
        self.com_img[:] = [255-a, 255-b, 255-c]
        self.col_img = self.col_img.astype('uint8')
        self.com_img = self.com_img.astype('uint8')
        
        background_img[:] = [255, 255, 255] 
        prediction_img[:]= [0,0,0]
        
        vix = np.where(self.pred[:,1] > self.doubleSpinBox.value())[0]
        
        
        for i in range(len(self.test_Z)):
            if i in vix:
                w = int(self.test_Z[i,2])
                h = int(self.test_Z[i,1])
                background_img[h,w] = [0,0,0]
                prediction_img[h,w] = [255,255,255]
                
        prediction_img = prediction_img.astype('uint8')
        background_img = background_img.astype('uint8')
        
        self.prediction_mask = prediction_img
        self.background_mask = background_img
        self.original_pred_mask=prediction_img
        self.original_back_mask=background_img
        self.radioButton_3.setChecked(True)
        self.make_prediction()
        self.make_result()
        self.setResult()

    def make_prediction(self):
        self.prediction_list = None

        a,b,c, = self.col
        self.col_img[:] = [self.col]
        self.com_img[:] = [255-a, 255-b, 255-c]
        self.col_img = self.col_img.astype('uint8')
        self.com_img = self.com_img.astype('uint8')
        a = cv2.subtract(self.col_img, self.background_mask)
        b = cv2.subtract(self.com_img, self.prediction_mask)
        self.prediction_list = cv2.add(a,b)

    def make_result(self):
        self.result_list = None
        a,b,c, = self.col
        self.col_img[:] = [self.col]
        self.com_img[:] = [255-a, 255-b, 255-c]
        self.col_img = self.col_img.astype('uint8')
        self.com_img = self.com_img.astype('uint8')
        a = cv2.subtract(self.col_img, self.background_mask)
        b = cv2.subtract(self.cropped_im, self.prediction_mask)
        self.result_list=cv2.add(a,b)
        
        
    def setPrediction(self):
        self.setPhoto2(self.prediction_list)


    def setResult(self):
        self.setPhoto2(self.result_list)

    #%% fx - color
      
    def color_value_now(self, name):
        a = list(name)
        del a[0]
        
        r = (self.hex_to_rgb(a[0])*16) + (self.hex_to_rgb(a[1]))
            
        g = (self.hex_to_rgb(a[2])*16) + (self.hex_to_rgb(a[3]))
        b = (self.hex_to_rgb(a[4])*16) + (self.hex_to_rgb(a[5]))
        return b,g,r



    def hex_to_rgb(self,t):
        if t == ('a'):
            y = 10
            return int(y)
        elif t == ('b'):
            y = 11
            return int(y)
        elif t == ('c'):
            y = 12
            return int(y)
        elif t == ('d'):
            y = 13
            return int(y)
        elif t == ('e'):
            y = 14
            return int(y)
        elif t == ('f'):
            y= 15
            return int(y)
        else: 
            y = t
            return int(y)
    

    def showDialog(self):
        col = QColorDialog.getColor()

        if col.isValid():
            b = col.name()
            self.col = self.color_value_now(b)
            self.select_col()
            
    def showDialog2(self):
        col = QColorDialog.getColor()

        if col.isValid():
            b = col.name()
            self.col2 = self.color_value_now(b)
            self.select_col2()
            
    def select_col(self):
        a, b, c = self.col
        self.col_img[:] = self.col
        self.com_img[:] = [255-a, 255-b, 255-c]
        if self.dots == []:
            self.make_prediction()
            self.make_result()
            if self.radioButton.isChecked():
                self.setPhoto2(self.cropped_im)
            elif self.radioButton_2.isChecked():
                self.setPrediction()
            elif self.radioButton_3.isChecked():
                self.setResult()
        else:
            self.confirm()
                
    def select_col2(self):
        img_color = self.finalprediction.copy()
        img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
        ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        img_color = self.cropped_im.copy()

        self.h = self.nosize.shape[0]*2**self.win_size
        img = imutils.resize(img_color, height=int(self.h))        

        for i in range(len(self.center)):
            cv.circle(img, self.mul_centers(2**(self.win_size),self.center)[i], 2*(2**self.win_size), self.col2, -1)       
        self.cellcounting = img
        self.result_list = self.cellcounting
        self.setPhoto2(self.cellcounting)                
            


    #%% fx - select roi
    def drawRects(self):

        oldx = oldy = -1 # default x y
        img = self.result_list.copy()
        self.h = img.shape[0]*2**self.win_size
        img = imutils.resize(img, height = int(self.h))
        
        self.dots = []

        
        mul = 2**self.win_size
        def on_mouse(event, x, y, flags, param):

            global oldx, oldy
            global dots

        
            if event == cv2.EVENT_LBUTTONDOWN: 
                oldx, oldy = x, y 
                self.dots.append([int(oldx/(2**self.win_size)), int(oldy/(2**self.win_size))])
                cv2.line(img, (x,y), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
        
            elif event == cv2.EVENT_LBUTTONUP:
                cv2.line(img,(int(self.dots[-1][0]*mul), int(self.dots[-1][1]*mul)), (x, y),  (0, 0, 255), 4, cv2.LINE_AA)
                if len(self.dots)<2:
                    pass
                else:
                    cv2.line(img,(int(self.dots[-2][0]*mul), int(self.dots[-2][1]*mul)), (int(self.dots[-1][0]*mul), int(self.dots[-1][1]*mul)),  (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow('', img)
                cv2.moveWindow('', self.winx, self.winy)
        
        
            elif event == cv2.EVENT_MOUSEMOVE: 
                if len(self.dots) == 0:
                    pass
                else:
                    b= self.dots.copy()
                    b.pop()
                    if [int(oldx/(2**self.win_size)), int(oldy/(2**self.win_size))] not in b:

                        draw=img.copy()
                        cv2.line(draw,(int(self.dots[-1][0]*mul), int(self.dots[-1][1]*mul)), (x, y),  (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('', draw)
                        cv2.moveWindow('', self.winx, self.winy)

        cv2.setMouseCallback('', on_mouse, img)
        
        

        cv2.imshow('', img)
        cv2.moveWindow('', self.winx, self.winy)
        cv2.waitKey()

        cv2.destroyAllWindows()
        self.confirm()


    #%% fx - confirm roi
    def confirm(self):

        self.make_prediction()
        self.make_result()
        
        white_color = (255,255,255)
        img = self.cropped_im

        a =np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        b = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        b[:,:] = white_color
        if len(self.dots) ==0:
            pass
        else:
            pt1 = np.array(self.dots, np.int32)
            a = cv2.fillPoly(a, [pt1], white_color) # ROI = white image
            b = cv2.subtract(b,a) # ROI = black image
        
            # original - mask (background for result (combined) image)
            c = cv2.subtract(self.cropped_im,a)
            
            # compl color image - mask (background for prediction (detected) image)
            e = cv2.subtract(self.com_img,a)
            
            # inside..
            for i in range(5):
                # prediction list - b + c
                d = cv2.subtract(self.prediction_list, b)
                self.prediction_list = cv2.add(d,e)
                
            for i in range(5):
                # result list - b + c
                d = cv2.subtract(self.result_list, b)
                self.result_list = cv2.add(d,c)        
            
            
            if self.radioButton.isChecked():
                self.setPhoto2(self.cropped_im)
            elif self.radioButton_2.isChecked():
                self.setPrediction()
            elif self.radioButton_3.isChecked():
                self.setResult()

            
    #%% fx - reset 1
    def reset(self):
        self.prediction_mask = self.original_pred_mask
        self.background_mask = self.original_back_mask
        self.make_prediction()
        self.make_result()
        self.setResult()
        
    def reset2(self):
        self.center = self.original_center.copy()
        self.settingstep3()
        self.points()


    #%% fx - setting step 3
    def settingstep3(self):
        self.label_status2.setText('cell counting')        
        self.status_now = 'cell counting'
        self.stop_waiting = 2
        item = self.doubleSpinBox.value()
        self.finalprediction = self.prediction_list      
        img_color = self.finalprediction.copy()
        img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
        ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        img_color = self.cropped_im.copy()
        
        mul = 2**self.win_size
        
        self.center = []
        if self.doubleSpinBox2.value() == 0:
            for cnt in contours:
                M = cv.moments(cnt)
                if M['m00'] == 0:
                    cx = int(cnt[0][0][0]/mul)
                    cy = int(cnt[0][0][1]/mul)
                    cv.circle(img_color, (int(cx*mul), int(cy*mul)), 2, (255,255,0), -1)
                    self.center.append((cx,cy))
                else:
                    cx = int(M['m10']/M['m00']/mul)
                    cy = int(M['m01']/M['m00']/mul)
                    cv.circle(img_color, (int(cx*mul), int(cy*mul)), 2, self.col2, -1)
                    self.center.append((cx,cy))                        
                        
        else:
            for cnt in contours:
                M = cv.moments(cnt)
                if M['m00'] <= self.doubleSpinBox2.value():
                    pass
                else: 
            
                    cx = int(M['m10']/M['m00']/mul)
                    cy = int(M['m01']/M['m00']/mul)
                    cv.circle(img_color, (int(cx*mul), int(cy*mul)), 2, self.col2, -1)
                    self.center.append((cx,cy))                

        self.original_center = self.center.copy()
        self.cellcounting = img_color
        self.setPhoto2(self.cellcounting)
        self.label_valuecounts.setText(str(len(self.center)))
        self.result_list = self.cellcounting
        

    #%% fx - center lists for different sizes (zoom)
    def mul_centers(self,mul,lst):
        mul_list = []
        ar = np.array(lst)
        for cor in lst:
            a = int(cor[0]*mul)
            b = int(cor[1]*mul)
            mul_list.append((a,b))
        return mul_list

    #%% fx - setting step 3 - button enabled
    def settingbutton3(self):
        self.stop_waiting = 2
        self.groupBox_celldetection.setEnabled(True)
        self.pushButton_next2.setEnabled(False)
        self.pushButton_color2.setEnabled(True)
        self.doubleSpinBox2.setEnabled(True)
        self.groupBox_prediction.setEnabled(False)
        self.pushButton_next3.setEnabled(True)

    #%% fx - setting last stage
    def settingstep4(self):
        self.pushButton_reset2.setEnabled(True)
        self.pushButton_color2.setEnabled(False)
        self.doubleSpinBox2.setEnabled(False)
        self.pushButton_next3.setEnabled(False)
        self.predstart()
    #%% fx - drawpoints
    
    def points(self):
        self.center = self.original_center
        img = self.nosize.copy()
        self.h = self.nosize.shape[0]*2**self.win_size
        img = imutils.resize(img, height=int(self.h))
        mul = 2**(self.win_size)
        oldx = oldy = -1

        for i in range(len(self.center)) :
            q =2**(self.win_size)
            p =self.center
            cv.circle(img, self.mul_centers(q,p)[i], int(2*q), self.col2, -1)


        def on_mouse(event, x, y, flags, param):

           global oldx, oldy 
           global dots
        
               
           if (event == cv2.EVENT_LBUTTONDOWN): 
               oldx, oldy = x, y 
           elif event == cv2.EVENT_LBUTTONUP: 
               self.center.append((int(oldx/(2**self.win_size)),int(oldy/(2**self.win_size))))
               img = self.nosize.copy()
               self.h = self.nosize.shape[0]*2**self.win_size
               img = imutils.resize(img, height=int(self.h))
               for i in range(len(self.center)):
                   cv.circle(img, self.mul_centers(2**(self.win_size),self.center)[i], 2*(2**self.win_size), self.col2, -1) 
               
               self.result_img = img 
               cv2.imshow('', img)
               cv2.moveWindow('', self.winx, self.winy)
               self.label_valuecounts.setText(str(len(self.center)))
               
           elif (event == cv2.EVENT_RBUTTONDOWN):
               oldx, oldy = x, y
               
           elif event == cv2.EVENT_RBUTTONUP:
               for i in range(len(self.center)):
                   a = int(oldx/(2**self.win_size))
                   b = int(oldy/(2**self.win_size))
                   if i not in range(len(self.center)):
                       pass
                   elif  (self.center[i][0]-5<a< self.center[i][0]+5) and (self.center[i][1]-5<b<self.center[i][1]+5):
                       del self.center[i]
               
               else:
                   pass
               img = self.nosize.copy()
               self.h = self.nosize.shape[0]*2**self.win_size
               img = imutils.resize(img, height=int(self.h))
               
               for i in range(len(self.center)):
                   cv.circle(img, self.mul_centers(2**(self.win_size),self.center)[i], int(2*(2**self.win_size)), self.col2, -1) 
               cv2.imshow('', img)
               cv2.moveWindow('', self.winx, self.winy)
               reimg = self.nosize.copy()
               for i in range(len(self.center)):
                   cv.circle(reimg, self.center[i], 2, self.col2, -1)
               self.result_list = img
               
               self.label_valuecounts.setText(str(len(self.center)))
       
        
        cv2.setMouseCallback('', on_mouse, img)
        
        

        cv2.imshow('', img)
        cv2.moveWindow('', self.winx, self.winy)


    #%% fx - save x y
    def save_markers(self):
        if self.stop_waiting <2:
            pass
        else:
            filename = QFileDialog.getSaveFileName(filter="CSV(*.csv)")[0]
            csvfile = open(filename, 'w', newline = "")
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['X','Y'])
            for row in self.center:
                csvwriter.writerow(row)
                
            csvfile.close()
            self.label_status2.setText('saved')
            time.sleep(3)
            self.label_status2.setText(self.status_now)            
            
            
    #%% fx - save photo
    def savePhoto(self):

        if str(type(self.tmp)) == "<class 'NoneType'>":
            pass
            
        elif self.stop_waiting == 0:
            filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
            cv2.imwrite(filename,self.nosize, [cv2.IMWRITE_JPEG_QUALITY,100])           
          
            self.label_status2.setText('saved')
            time.sleep(3)
            self.label_status2.setText(self.status_now)
        
        elif self.stop_waiting == 1:
            if self.radioButton.isChecked():
                save_img = self.nosize
            elif self.radioButton_2.isChecked():
                save_img = self.prediction_list
            elif self.radioButton_3.isChecked():
                save_img = self.result_list

            filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
            cv2.imwrite(filename,save_img, [cv2.IMWRITE_JPEG_QUALITY,100])           
            self.label_status2.setText('saved')
            time.sleep(3)
            self.label_status2.setText(self.status_now)
        
        elif self.stop_waiting == 2:
            if self.radioButton.isChecked():
                save_img = self.nosize
            elif self.radioButton_2.isChecked():
                save_img = self.prediction_list
            elif self.radioButton_3.isChecked():
                img = self.nosize.copy()
                self.h = self.nosize.shape[0]*2**self.win_size
                save_img = imutils.resize(img, height=int(self.h))
                for i in range(len(self.center)):
                    cv.circle(save_img, self.mul_centers(2**(self.win_size),self.center)[i], int(2*(2**self.win_size)), self.col2, -1) 
            filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
            cv2.imwrite(filename,save_img, [cv2.IMWRITE_JPEG_QUALITY,100])           
            self.label_status2.setText('saved')
            time.sleep(3)
            self.label_status2.setText(self.status_now)
        
        
        elif self.stop_waiting == 3:
            if self.radioButton.isChecked():
                save_img = self.nosize
            elif self.radioButton_2.isChecked():
                save_img = self.prediction_list
            elif self.radioButton_3.isChecked():
                img = self.nosize.copy()
                self.h = self.nosize.shape[0]*2**self.win_size
                save_img = imutils.resize(img, height=int(self.h))
                for i in range(len(self.center)):
                    cv.circle(save_img, self.mul_centers(2**(self.win_size),self.center)[i], int(2*(2**self.win_size)), self.col2, -1) 
            filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
            cv2.imwrite(filename,save_img, [cv2.IMWRITE_JPEG_QUALITY,100])           
            self.label_status2.setText('saved')
            time.sleep(3)
            self.label_status2.setText(self.status_now)                
                    
            
    #%% fx - extract filename 
    def name(self, f):
        s = f
        s = s.split('/')
        s = s[-1]
        s = s.split('.')
        s = s[0]  
        return s
    #%% fx - move window
    def switch(self):
        while True:
            key = cv.waitKey()

            ## move                    
            if (key == ord('a')) or (key == ord('A')):
                self.winx += -20
                self.setPhoto2(self.tmp2)

            if (key == ord('d')) or (key == ord('D')):
                self.winx += 20
                self.setPhoto2(self.tmp2)

            if (key == ord('w')) or (key == ord('W')):
                self.winy += -20
                self.setPhoto2(self.tmp2)
                
                        
            if (key == ord('s')) or (key == ord('S')):
                self.winy += 20
                self.setPhoto2(self.tmp2)  
                
            key = cv2.waitKeyEx()  
            if key == 0x260000:
                if -5<self.win_size<4:
                    self.win_size += 1
                    self.setPhoto2(self.tmp2)
                
            elif (key == 0x280000):
                if -4<self.win_size<5:
                    self.win_size +=  -1
                    self.setPhoto2(self.tmp2)
                    
            if self.stop_waiting == 2:
                break
            
            if cv2.getWindowProperty('', cv2.WND_PROP_VISIBLE) <1:
                break     
                

        cv2.destroyAllWindows()                 
    #%% fx - switch image mode
    def mode(self):
        while True:
            key = cv.waitKey()

            if (key == ord('z')) or (key == ord('Z')):
                self.setPhoto2(self.cropped_im)
                self.radioButton.setChecked(True)
                
            if (key == ord('x')) or (key == ord('X')):
                self.setPhoto2(self.prediction_list)
                self.radioButton_2.setChecked(True)

            if (key == ord('c')) or (key == ord('C')):
                self.setPhoto2(self.result_list)
                self.radioButton_3.setChecked(True)
                
            if (key == ord('a')) or (key == ord('A')):
                self.winx += -20
                self.setPhoto2(self.tmp2)

            elif (key == ord('d')) or (key == ord('D')):
                self.winx += 20
                self.setPhoto2(self.tmp2)

            if (key == ord('w')) or (key == ord('W')):
                self.winy += -20
                self.setPhoto2(self.tmp2)
                
                        
            elif (key == ord('s')) or (key == ord('S')):
                self.winy += 20
                self.setPhoto2(self.tmp2) 
                
            key = cv2.waitKeyEx()  
            if key == 0x260000:
                if -5<self.win_size<4:
                    self.win_size += 1
                    self.setPhoto2(self.tmp2)
                
            elif (key == 0x280000):
                if -4<self.win_size<5:
                    self.win_size +=  -1
                    self.setPhoto2(self.tmp2)
                    
            if cv2.getWindowProperty('', cv2.WND_PROP_VISIBLE) <1:
                break     
            if self.stop_waiting == 3:
                break

        cv.destroyAllWindows()            

    #%% fx - last modi
    def lastmodi(self):
        while True:
            
            key = cv.waitKey()
            if (key == ord('z')) or (key == ord('Z')):
                self.setPhoto2(self.cropped_im)
                self.radioButton.setChecked(True)
                
            elif (key == ord('x')) or (key == ord('X')):
                self.setPhoto2(self.prediction_list)
                self.radioButton_2.setChecked(True)

            elif (key == ord('c')) or (key == ord('C')):
                self.setPhoto3()
                self.radioButton_3.setChecked(True)
                
            elif (key == ord('a')) or (key == ord('A')):
                self.winx += -20
                self.setPhoto3()

            elif (key == ord('d')) or (key == ord('D')):
                self.winx += 20
                self.setPhoto3()

            elif (key == ord('w')) or (key == ord('W')):
                self.winy += -20
                self.setPhoto3()
                
                        
            elif (key == ord('s')) or (key == ord('S')):
                self.winy += 20
                self.setPhoto3() 
                
            key = cv2.waitKeyEx()  
            if key == 0x260000:
                if -5<self.win_size<4:
                    self.win_size += 1
                    self.setPhoto3()
                
            elif (key == 0x280000):
                if -4<self.win_size<5:
                    self.win_size +=  -1
                    self.setPhoto3()
                    
            if cv2.getWindowProperty('', cv2.WND_PROP_VISIBLE) <1:
                break     


        cv.destroyAllWindows() 

       
        


#%% retranslate
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "UI"))
        self.pushButton_next1.setStatusTip(_translate("MainWindow", "Start detection"))
        self.pushButton_next1.setText(_translate("MainWindow", "▶"))
        self.groupBox_BC.setStatusTip(_translate("MainWindow", "Manual adjustment"))
        self.groupBox_BC.setTitle(_translate("MainWindow", "Brightness & Contrast"))
        self.label_valueC.setText(_translate("MainWindow", "1"))
        self.horizontalSlider_C.setStatusTip(_translate("MainWindow", "Adjust contrast"))
        self.horizontalSlider_B.setStatusTip(_translate("MainWindow", "Adjust brightness"))
        self.label_C.setText(_translate("MainWindow", "C"))
        self.label_valueB.setText(_translate("MainWindow", "0"))
        self.label_B.setText(_translate("MainWindow", "B"))
        self.groupBox_crop.setStatusTip(_translate("MainWindow", "Select a rectangular area including SNc"))
        self.groupBox_crop.setTitle(_translate("MainWindow", "Crop"))
        self.pushButton_crop.setText(_translate("MainWindow", "□"))
        self.groupBox_prediction.setTitle(_translate("MainWindow", "Cell Detection"))
        self.doubleSpinBox.setStatusTip(_translate("MainWindow", "Lower the level is larger the predicted region will be"))
        self.pushButton_color1.setStatusTip(_translate("MainWindow", "Change the color of the predicted cell regions"))
        self.pushButton_color1.setText(_translate("MainWindow", "color"))
        self.label_level.setText(_translate("MainWindow", "level"))
        self.pushButton_select.setStatusTip(_translate("MainWindow", "Click points to draw a polygon. Cells outside the polygon will be excluded from the counting"))
        self.pushButton_select.setText(_translate("MainWindow", "select"))
        self.pushButton_reset1.setStatusTip(_translate("MainWindow", "Reset detection result"))
        self.pushButton_reset1.setText(_translate("MainWindow", "reset"))
        self.groupBox_celldetection.setTitle(_translate("MainWindow", "Cell Counting"))
        self.pushButton_color2.setStatusTip(_translate("MainWindow", "Change the color of the markers"))
        self.pushButton_color2.setText(_translate("MainWindow", "color"))
        self.label_size.setText(_translate("MainWindow", "size"))
        self.doubleSpinBox2.setStatusTip(_translate("MainWindow", "Set a cell size threshold: detected areas whose sizes exceed the threshold are retained."))
        self.pushButton_next3.setStatusTip(_translate("MainWindow", "Save previous changes and add or delete markers. Click left button to add and double-click right button to delete markers."))
        self.pushButton_next3.setText(_translate("MainWindow", "▶"))
        self.pushButton_reset2.setStatusTip(_translate("MainWindow", "Reset result of manual addition and elimination of markers"))
        self.pushButton_reset2.setText(_translate("MainWindow", "reset"))
        self.pushButton_next2.setStatusTip(_translate("MainWindow", "Start cell counting"))
        self.pushButton_next2.setText(_translate("MainWindow", "▶"))
        self.groupBox_imshow.setTitle(_translate("MainWindow", "Image Show"))
        self.radioButton.setStatusTip(_translate("MainWindow", "Original image"))
        self.radioButton.setText(_translate("MainWindow", "Original (Z)"))
        self.radioButton.setShortcut(_translate("MainWindow", "Z"))
        self.radioButton_2.setStatusTip(_translate("MainWindow", "detection result"))
        self.radioButton_2.setText(_translate("MainWindow", "Detected (X)"))
        self.radioButton_2.setShortcut(_translate("MainWindow", "X"))
        self.radioButton_3.setText(_translate("MainWindow", "Combined (C)"))
        self.radioButton_3.setShortcut(_translate("MainWindow", "C"))
        self.label_status2.setText(_translate("MainWindow", "Open an image file to get started"))
        self.label_counts.setText(_translate("MainWindow", "Counts:"))
        self.label_valuecounts.setText(_translate("MainWindow", "0"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSave.setTitle(_translate("MainWindow", "Save"))        
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionImage.setText(_translate("MainWindow", "Image"))
        self.actionImage.setShortcut(_translate("MainWindow", "Ctrl+S"))
        
        self.actionMarkers.setText(_translate("MainWindow", "Markers"))
        self.actionMarkers.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.actionUser_Guide.setText(_translate("MainWindow", "User Guide"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

