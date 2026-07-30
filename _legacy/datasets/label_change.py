# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:03:55 2025

@author: Administrator
"""
import numpy as np


def khartoum_1(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {3,15, 35, 41, 42,47,48,49}
    # print("处理前标签唯一值: ", np.unique(loc_labels))
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 3)] = 1
    loc_labels[(loc_label_orig == 15)] = 2
    loc_labels[(loc_label_orig == 35)] = 3
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43)] = 4
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)] = 5

    
    
    clf_labels[(clf_label_orig == 35) | (clf_label_orig == 41) | (clf_label_orig == 47)] = 1
    # clf_label[(clf_label_orig == 35) | (clf_label_orig == 41)] = 1
    
    clf_labels[(clf_label_orig == 3) | (clf_label_orig == 15) | (clf_label_orig == 42)| (clf_label_orig == 48)] = 2
    
    # print("处理后标签唯一值: ", np.unique(loc_labels))
    clf_labels[~np.isin(clf_labels, [1, 2])] = 255
    return loc_labels,clf_labels

def omdurman_1(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {26, 27, 35, 36, 41, 42}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
    # print("处理标签唯一值: ", np.unique(loc_label))
        
    loc_labels[(loc_label_orig == 26) | (loc_label_orig == 27)] = 1
    loc_labels[(loc_label_orig == 35) | (loc_label_orig == 36)] = 2
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42)] = 3
        
        
    clf_labels[(clf_label_orig == 26) | (clf_label_orig == 35)| (clf_label_orig == 41)] = 1
    clf_labels[(clf_label_orig == 27) | (clf_label_orig == 36)| (clf_label_orig == 42)] = 2
        
    # print("处理标签唯一值: ", np.unique(loc_label))
    clf_labels[~np.isin(clf_labels, [1, 2])] = 255
    return loc_labels,clf_labels

def melitopol_airport(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {2,8,14,26,41,42,43,47,48,49,50}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 2)] = 1
    loc_labels[(loc_label_orig == 8)] = 2
    loc_labels[(loc_label_orig == 14)] = 3
    loc_labels[(loc_label_orig == 26)] = 4
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43)] = 5
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)| (loc_label_orig == 50)] = 6

    clf_labels[(clf_label_orig == 2) | (clf_label_orig == 8)| (clf_label_orig == 14) | (loc_label_orig == 26) | 
               (loc_label_orig == 41) | (loc_label_orig == 43) | (loc_label_orig == 47)| (loc_label_orig == 49)| (loc_label_orig == 50)] = 1
    clf_labels[(clf_label_orig == 42) | (clf_label_orig == 48)] = 2
        
    clf_labels[~np.isin(clf_labels, [1, 2])] = 255

    return loc_labels,clf_labels

def test_all(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {2,3,8,9,14,15,16,17,26,27,35,36,41,42,43,44,47,48,49,50,77,78,79,80,105}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 2) | (loc_label_orig == 3)] = 1
    loc_labels[(loc_label_orig == 8) | (loc_label_orig == 9)] = 2
    loc_labels[(loc_label_orig == 14) | (loc_label_orig == 15) | (loc_label_orig == 16) | (loc_label_orig == 17)] = 3
    loc_labels[(loc_label_orig == 26) | (loc_label_orig == 27)] = 4
    loc_labels[(loc_label_orig == 35) | (loc_label_orig == 36)] = 5
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43)] = 6
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)| (loc_label_orig == 50)] = 7
    loc_labels[(loc_label_orig == 77) | (loc_label_orig == 78) | (loc_label_orig == 79)| (loc_label_orig == 80)] = 8
    
    loc_labels[(loc_label_orig == 105)] = 9
    
    loc_labels[(loc_label_orig == 106)] = 10
    
    # no-change
    clf_labels[(clf_label_orig == 2) | (clf_label_orig == 8)| (clf_label_orig == 14) | (loc_label_orig == 26) 
               | (clf_label_orig == 35) 
               | (clf_label_orig == 41) 
               | (clf_label_orig == 47) 
               | (clf_label_orig == 77) ] = 1
    # damaged
    clf_labels[(clf_label_orig == 3) | (clf_label_orig == 9) | (clf_label_orig == 15) 
               | (clf_label_orig == 27) | (clf_label_orig == 36) | (clf_label_orig == 42) | (clf_label_orig == 48)
               | (clf_label_orig == 78) ] = 2
    # added
    clf_labels[(clf_label_orig == 17) | (clf_label_orig == 44)| (clf_label_orig == 50) | (clf_label_orig == 80) | (clf_label_orig == 105) | (clf_label_orig == 106)] = 3   
    
    # reduced
    clf_labels[(clf_label_orig == 16) | (clf_label_orig == 43)| (clf_label_orig == 49) | (clf_label_orig == 79)] = 4
        
    clf_labels[~np.isin(clf_labels, [1, 2, 3, 4])] = 255
    return loc_labels,clf_labels

def building(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {41,42,43}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43)] = 1

    
    clf_labels[(clf_label_orig == 42)| (clf_label_orig == 43)] = 1
    
    # print("处理后标签唯一值: ", np.unique(loc_labels))
    clf_labels[~np.isin(clf_labels, 1)] = 0
    
    return loc_labels,clf_labels

def plane(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {47,48,49,50,51}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)| (loc_label_orig == 50)| (loc_label_orig == 51)] = 1

    
    clf_labels[(clf_label_orig == 47)| (loc_label_orig == 49)| (loc_label_orig == 50)| (loc_label_orig == 51)] = 1    
    clf_labels[(clf_label_orig == 48)] = 2
    
    # print("处理后标签唯一值: ", np.unique(loc_labels))
    clf_labels[~np.isin(clf_labels, [1, 2])] = 255
    
    return loc_labels,clf_labels


def port(loc_label, clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    
    # 更新 keep_values 以包含所有可能的值
    keep_values = {20,21,41,42,43,83,84,85,89,90,95,96}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    # 映射 loc_label 到类别索引
    loc_labels[(loc_label_orig == 20) | (loc_label_orig == 21)| (loc_label_orig == 83) | (loc_label_orig == 84) | (loc_label_orig == 85)] = 1  # 码头/浮船坞
    
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43) | (loc_label_orig == 95) | (loc_label_orig == 96)] = 2   # 建筑物/大型罐体
    
    # loc_labels[(loc_label_orig == 83) | (loc_label_orig == 84) | (loc_label_orig == 85)] = 3 # 浮船坞
    
    loc_labels[(loc_label_orig == 89) | (loc_label_orig == 90) ] = 3  # 栈桥
    
    # loc_labels[(loc_label_orig == 95) | (loc_label_orig == 96) ] = 5  # 大型罐体

    # 映射 clf_label 到变化类型索引（只使用 clf_label_orig）
    clf_labels[(clf_label_orig == 20) | (clf_label_orig == 41) | 
               (clf_label_orig == 43) | (clf_label_orig == 40) | 
               (clf_label_orig == 41) | (clf_label_orig == 83) | (clf_label_orig == 85) | 
               (clf_label_orig == 89) | (clf_label_orig == 95)] = 1  # 无变化
    
    clf_labels[(clf_label_orig == 21) |  (clf_label_orig == 42) | 
               (clf_label_orig == 84) | (clf_label_orig == 90) | (clf_label_orig == 96)] = 2  # 损伤
    # clf_labels[(clf_label_orig == 44)] = 3  # 新建
    # clf_labels[(clf_label_orig == 45)] = 4  # 扩建
    # clf_labels[(clf_label_orig == 16) | (clf_label_orig == 43)] = 5  # 拆除

    # 将不在 [1,2,3,4,5] 的值设为 255（背景或无效）
    # clf_labels[~np.isin(clf_labels, [1,2,3])] = 255
    clf_labels[~np.isin(clf_labels, [1,2])] = 255
    
    return loc_labels, clf_labels


def airport(loc_label, clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    
    # 更新 keep_values 以包含所有可能的值
    keep_values = {2, 3, 8, 9, 14, 15, 16, 40, 41, 42, 43, 44, 45, 52, 53, 54, 65, 66, 71, 72}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    # 映射 loc_label 到类别索引
    # loc_labels[(loc_label_orig == 2) | (loc_label_orig == 3) | (loc_label_orig == 8) | (loc_label_orig == 9)] = 1  # 跑道/滑道
    loc_labels[(loc_label_orig == 2) | (loc_label_orig == 3)] = 1  # 跑道
    loc_labels[(loc_label_orig == 8) | (loc_label_orig == 9)] = 2   # 滑道
    loc_labels[(loc_label_orig == 14) | (loc_label_orig == 15) | (loc_label_orig == 16)] = 3 # 停机坪
    loc_labels[(loc_label_orig == 40) | (loc_label_orig == 41) | (loc_label_orig == 42)
              | (loc_label_orig == 43) | (loc_label_orig == 44) | (loc_label_orig == 45)
              | (loc_label_orig == 65) | (loc_label_orig == 66)
              | (loc_label_orig == 71) | (loc_label_orig == 72)
              | (loc_label_orig == 52) | (loc_label_orig == 53) | (loc_label_orig == 54)] = 4  # 建筑

    # 映射 clf_label 到变化类型索引（只使用 clf_label_orig）
    clf_labels[(clf_label_orig == 2) | (clf_label_orig == 8) | 
               (clf_label_orig == 14) | (clf_label_orig == 40) | 
               (clf_label_orig == 41) | (clf_label_orig == 52) | (clf_label_orig == 53) | 
               (clf_label_orig == 65) | (clf_label_orig == 71)] = 1  # 无变化
    
    clf_labels[(clf_label_orig == 3) |  (clf_label_orig == 9) | (clf_label_orig == 15) | (clf_label_orig == 42) | (clf_label_orig == 54) | 
               (clf_label_orig == 66)| (clf_label_orig == 72)] = 2  # 损伤
    # clf_labels[(clf_label_orig == 44)] = 3  # 新建
    # clf_labels[(clf_label_orig == 45)] = 4  # 扩建
    # clf_labels[(clf_label_orig == 16) | (clf_label_orig == 43)] = 5  # 拆除

    # 将不在 [1,2,3,4,5] 的值设为 255（背景或无效）
    # clf_labels[~np.isin(clf_labels, [1,2,3])] = 255
    clf_labels[~np.isin(clf_labels, [1,2])] = 255
    
    return loc_labels, clf_labels

def airport_add(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {2,3,8,9,14,15,16,41,42,43,47,48,49,50,53,54,65,66,71,72}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 2) | (loc_label_orig == 3)] = 1
    loc_labels[(loc_label_orig == 8) | (loc_label_orig == 9)] = 2
    loc_labels[(loc_label_orig == 14) | (loc_label_orig == 15)] = 3
    # loc_labels[(loc_label_orig == 26) | (loc_label_orig == 27)] = 4
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43) 
               | (loc_label_orig == 65) | (loc_label_orig == 66)
               | (loc_label_orig == 71) | (loc_label_orig == 52) | (loc_label_orig == 53) | (loc_label_orig == 54)] = 4
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)| (loc_label_orig == 50)] = 5
    # loc_labels[(loc_label_orig == 65) | (loc_label_orig == 66)] = 6
    # loc_labels[(loc_label_orig == 71) | (loc_label_orig == 72)] = 6

    
    clf_labels[(clf_label_orig == 2) | (clf_label_orig == 8)| (clf_label_orig == 14) 
               | (loc_label_orig == 41) | (loc_label_orig == 43) 
               | (loc_label_orig == 52) | (loc_label_orig == 53) | (loc_label_orig == 65)
               | (loc_label_orig == 71) | (loc_label_orig == 77) | (loc_label_orig == 79)
               | (loc_label_orig == 47)| (loc_label_orig == 49)| (loc_label_orig == 50)] = 1   
    
    clf_labels[(clf_label_orig == 3) | (clf_label_orig == 9) | (clf_label_orig == 15) 
               | (clf_label_orig == 42) 
               | (clf_label_orig == 54) | (clf_label_orig == 66)| (loc_label_orig == 48)
               | (clf_label_orig == 72) | (loc_label_orig == 78)] = 2
    
    # print("处理后标签唯一值: ", np.unique(loc_labels))
    clf_labels[~np.isin(clf_labels, [1, 2])] = 0
    
    return loc_labels,clf_labels


def test_damage(loc_label,clf_label):
    clf_labels = clf_label.copy()
    loc_labels = loc_label.copy()
    clf_label_orig = clf_label.copy()
    loc_label_orig = loc_label.copy()
    keep_values = {2,3,8,9,14,15,16,17,26,27,35,36,41,42,43,44,47,48,49,50,77,78,79,80,105}
    loc_labels[~np.isin(loc_label, keep_values)] = 0
    clf_labels[~np.isin(clf_label, keep_values)] = 0
        
    loc_labels[(loc_label_orig == 2) | (loc_label_orig == 3)] = 1
    loc_labels[(loc_label_orig == 8) | (loc_label_orig == 9)] = 2
    loc_labels[(loc_label_orig == 14) | (loc_label_orig == 15) | (loc_label_orig == 16) | (loc_label_orig == 17)] = 3
    loc_labels[(loc_label_orig == 26) | (loc_label_orig == 27)] = 4
    loc_labels[(loc_label_orig == 35) | (loc_label_orig == 36)] = 5
    loc_labels[(loc_label_orig == 41) | (loc_label_orig == 42) | (loc_label_orig == 43)] = 6
    loc_labels[(loc_label_orig == 47) | (loc_label_orig == 48) | (loc_label_orig == 49)| (loc_label_orig == 50)] = 7
    loc_labels[(loc_label_orig == 77) | (loc_label_orig == 78) | (loc_label_orig == 79)| (loc_label_orig == 80)] = 8
    
    loc_labels[(loc_label_orig == 105)] = 9
    
    loc_labels[(loc_label_orig == 106)] = 10
    
    # no-change
    clf_labels[(clf_label_orig == 2) | (clf_label_orig == 8)| (clf_label_orig == 14) | (loc_label_orig == 26) 
               | (clf_label_orig == 35) 
               | (clf_label_orig == 41) 
               | (clf_label_orig == 47) 
               | (clf_label_orig == 77) ] = 1
    # damaged
    clf_labels[(clf_label_orig == 3) | (clf_label_orig == 9) | (clf_label_orig == 15) 
               | (clf_label_orig == 27) | (clf_label_orig == 36) | (clf_label_orig == 42) | (clf_label_orig == 48)
               | (clf_label_orig == 78) ] = 2

        
    clf_labels[~np.isin(clf_labels, [1, 2])] = 255
    return loc_labels,clf_labels



