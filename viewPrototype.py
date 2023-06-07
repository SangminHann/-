import streamlit as st
import cv2
import numpy as np
import os
import findQuesion as fq
st.set_page_config(layout="wide", page_title="오답의 정석")

empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con2, con3, empty2 = st.columns([0.5, 0.14, 0.86, 0.5])
empyt1, tmp1, con4, tmp2, empty2 = st.columns([0.3, 0.2, 0.6, 0.2, 0.3])
empyt1, tmp1, button_con, check_con, tmp2, empty2 = st.columns([0.3, 0.2, 0.3, 0.3, 0.2, 0.3])
empyt1, upload_con, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con5, empty2 = st.columns([0.3, 1.0, 0.3])

display_logo = True
directory = './test/'
p_open_tag = '<p style="font-size: 25px;"><b>'
p_close_tag = '</b></p>'

if display_logo is True:
    with con1:
        st.image('./HCI_PARTS/img/logo.png')
    with con3:
        genre = st.radio("함수선택", ['사진', 'PDF 변환 파일'], label_visibility = 'hidden', horizontal= True)
        
    with con4:
        file_name = st.file_uploader('채점된 문제 파일을 선택해주세요', type=['png', 'jpg', 'jpeg'])
    
    with button_con:
        is_button = st.button('오답노트 만들기')
        
    with check_con:
        check = False
        check = st.checkbox(label = '업로드 파일 보기')
        
    if file_name is not None and check is True:
        with upload_con:
            st.image(directory + file_name.name)
    
    if file_name is not None and check is True:
        with upload_con:
            st.text('업로드 된 파일이 없습니다.')
    
    if is_button and genre is None:
        with con3:
            st.markdown(p_open_tag + '파일 유형을 선택해 주세요' + p_close_tag, unsafe_allow_html = True)
            
    if is_button : 
        if file_name is not None:
            display_logo = False
                
        else:
            with con4:
                st.text('선택된 문제가 없습니다')
            is_button = False
                
if display_logo is False and genre == 'PDF 변환 파일':
    file_dir = './test/wrong/'
    fq.deleteAllFiles(file_dir)
    num = fq.trimWrongImg(directory + file_name.name)
    
    if num == 0:
        with con4:
            st.markdown(p_open_tag + '틀린 문제가 없습니다.' + p_close_tag, unsafe_allow_html = True)
            
    if num > 0:
        prefix = 'wrong_'
        suffix = '.png'
        tab_name = []
        name = '오답노트 '
        
        for i in range(num):
            tab_name.append(name + str(i + 1))
            
        with con5:
            t_key = 1
            tab = st.tabs(tab_name)
            
            for i in range(num):
                with tab[i]:
                    st.markdown(p_open_tag + '틀린 문제' + p_close_tag, unsafe_allow_html = True)
                    st.image(file_dir + prefix + str(i + 1) + suffix)
                    tmp1, tmp2 = st.columns([0.5,0.5])
                    
                    with tmp1:
                        st.markdown(p_open_tag + '틀린이유' + p_close_tag, unsafe_allow_html = True)
                        st.text_area('틀린이유', placeholder = "틀린 이유를 입력하세요", height=350, label_visibility = 'collapsed',key = t_key)
                        t_key += 1
                        
                    with tmp2:
                        st.markdown(p_open_tag + '문제 풀이' + p_close_tag, unsafe_allow_html = True)
                        st.text_area('틀린이유', placeholder = "풀이를 작성하세요", height=350, label_visibility = 'collapsed', key = t_key)
                        t_key += 1
                        
                if i == num - 1:  
                    while(True):
                            break_button = None
                            
if display_logo is False and genre == '사진':
    file_dir = './test/wrong/'
    fq.deleteAllFiles(file_dir)
    num = fq.trimWrongPic(directory + file_name.name)
    
    if num == 0:
        with con2:
            st.markdown(p_open_tag + '틀린 문제가 없습니다.' + p_close_tag, unsafe_allow_html = True)
            
    if num > 0:
        prefix = 'wrong_'
        suffix = '.png'
        tab_name = []
        name = '오답노트 '
        
        for i in range(num):
            tab_name.append(name + str(i + 1))
            
        with con5:
            t_key = 1
            tab = st.tabs(tab_name)
            
            for i in range(num):
                with tab[i]:
                    st.markdown(p_open_tag + '틀린 문제' + p_close_tag, unsafe_allow_html = True)
                    st.image(file_dir + prefix + str(i + 1) + suffix)
                    tmp1, tmp2 = st.columns([0.5,0.5])
                    
                    with tmp1:
                        st.markdown(p_open_tag + '틀린이유' + p_close_tag, unsafe_allow_html = True)
                        st.text_area('틀린이유', placeholder = "틀린 이유를 입력하세요", height=350, label_visibility = 'collapsed',key = t_key)
                        t_key += 1
                        
                    with tmp2:
                        st.markdown(p_open_tag + '문제 풀이' + p_close_tag, unsafe_allow_html = True)
                        st.text_area('틀린이유', placeholder = "풀이를 작성하세요", height=350, label_visibility = 'collapsed', key = t_key)
                        t_key += 1
                        
                if i == num - 1:  
                    while(True):
                            break_button = None