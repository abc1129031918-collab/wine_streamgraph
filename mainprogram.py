import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import sys
import os
import colorsys
import matplotlib.patheffects as path_effects
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import requests
import io
import unicodedata
import re
import math
import threading
import time

matplotlib.use('TkAgg') 

def resource_path(relative_path):
    """실행 파일(.exe)과 같은 위치에 있는 외부 폴더/파일 경로를 반환"""
    if getattr(sys, 'frozen', False):
        # .exe로 실행 중일 때: .exe 파일이 있는 실제 디렉토리 경로
        base_path = os.path.dirname(sys.executable)
    else:
        # 일반 .py로 실행 중일 때: 현재 소스 코드 폴더 경로
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def draw_wine_graph_on_frame(analyzer, wine_data, target_frame):
    # 1. 기존 위젯 제거
    for widget in target_frame.winfo_children():
        widget.destroy()
        
    wine_id = wine_data.get('id')
    if not wine_id: return

    # -----------------------------------------------------
    # [수정] 폴더 경로 안전하게 확보
    # -----------------------------------------------------
    raw_dir = resource_path("cleaned")
    data_dir = resource_path("data")
    
    # 폴더가 없으면 만듭니다. (이게 없어서 저장이 안 된 것임)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_file_path = os.path.join(raw_dir, f"wine_{wine_id}_clean.jsonl")
    data_file_path = os.path.join(data_dir, f"wine_{wine_id}_data.json")

    # 2. 데이터 파일 확인 및 생성
    if not os.path.exists(data_file_path):
        if os.path.exists(raw_file_path):
            # 로딩 메시지
            lbl_loading = tk.Label(target_frame, text="Analyzing reviews...", bg='#1e1e1e', fg='white')
            lbl_loading.pack(pady=20)
            target_frame.update()
            
            # 분석 실행
            success = analyzer.extract_and_save_data(raw_file_path, data_file_path)
            lbl_loading.destroy()
            
            if not success:
                tk.Label(target_frame, text="Analysis Failed.", bg='#1e1e1e', fg='red').pack()
                return
        else:
            tk.Label(target_frame, text="No review data.", bg='#1e1e1e', fg='gray', font=('Arial', 16)).pack(expand=True)
            return
    
    # 3. 그래프 그리기
    try:
        fig = analyzer.create_graph_from_data(data_file_path)
        
        canvas = FigureCanvasTkAgg(fig, master=target_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(background='#1e1e1e', highlightbackground='#1e1e1e')
        canvas_widget.pack(side='top', fill='both', expand=True)
        
        target_frame.update_idletasks()
        
        canvas_width_px = canvas_widget.winfo_width()
        canvas_height_px = canvas_widget.winfo_height()
        current_dpi = fig.get_dpi()
        
        if canvas_width_px > 10 and canvas_height_px > 10:
            fig.set_size_inches(canvas_width_px / current_dpi, canvas_height_px / current_dpi)
            
        canvas.draw()
        
    except Exception as e:
        tk.Label(target_frame, text=f"Error: {e}", bg='#1e1e1e', fg='red').pack()

class WineStreamAnalyzer:
    def __init__(self):
        # 1. 시간축 앵커
        self.section_anchors = {
            'nose': 0.1, 'aroma': 0.1, 'bouquet': 0.15, 'smell': 0.15, 'scent': 0.15, 
            'sniff': 0.15, 'opening': 0.15, 'color': 0.05, 'eye': 0.05, 
            'attack': 0.15, 'entry': 0.15, 'start': 0.1,
            'palate': 0.5, 'taste': 0.5, 'mouth': 0.5, 'flavor': 0.5, 'flavour': 0.5,
            'body': 0.45, 'texture': 0.45, 'mouthfeel': 0.45, 'mid': 0.5, 'middle': 0.5,
            'sip': 0.4, 'drink': 0.4, 'tongue': 0.5,
            'finish': 0.85, 'aftertaste': 0.9, 'end': 0.9, 'ending': 0.9, 
            'conclusion': 0.9, 'tail': 0.9, 'linger': 0.88, 'lingering': 0.88
        }

        # 2. 강도 수식어
        self.intensity_modifiers = {
            'hint': 0.2, 'hints': 0.2, 'touch': 0.2, 'trace': 0.2, 'whisper': 0.3,
            'subtle': 0.2, 'light': 0.3, 'faint': 0.2, 'delicate': 0.4, 'mild': 0.6, 'medium': 0.6,
            'slight': 0.3, 'slightly': 0.3, 'soft': 0.4, 'shy': 0.5, 'background': 0.4,
            'strong': 1.0, 'powerful': 1.0, 'bold': 1.0, 'intense': 1.0, 'deep': 1.0,
            'heavy': 1.0, 'rich': 1.0, 'concentrated': 1.0, 'pronounced': 1.0,
            'explosion': 1.0, 'bomb': 1.0, 'burst': 1.0, 'blast': 1.0,
            'dominant': 1.0, 'massive': 1.0, 'extreme': 1.0, 'super': 1.0,
            'very': 1.0, 'lots': 1.0, 'much': 1.0, 'full': 1.0,
            'big': 1.0, 'sharp': 1.0, 'good': 1.0, 'excellent':1.0, 'great':1.0, 'nice':1.0,
        }
        
        # 3. 아로마 휠 데이터베이스
        self.flavor_db = self._build_aroma_wheel_db()

        # [NEW] 계열별 강화 리스트 (Booster Families)
        # 키(Key) 단어가 많이 언급되면, 리스트 안의 맛(Flavor)들을 강화시킵니다.
        self.flavor_families = {
            'earthy': ['Mineral', 'Vegetal', 'Animal', 'Woods','Earthy'], # 흙내음은 미네랄, 식물성, 동물성, 나무 향을 모두 포함
            'fruity': ['Citrus', 'Pome Fruit', 'Stone Fruit', 'Tropical', 'Red Berries', 'Black Berries'], # 모든 과일 카테고리
            'red fruit': ['Red Berries'],
            'black fruit': ['Red Berries'],
            'ripe': ['Dried Fruit'],
            'floral': ['Floral'],
            'vegetality': ['Vegetal'], 
            'woody': ['Woods'],
            'malolactic': ['Malolactic', 'Yeast'], # 젖산 발효는 효모/빵 냄새와 연관됨
            'nutty': ['Nuts'],
            'toasty': ['Toasted', 'Spice'], # 토스트는 오크 숙성 스파이스와 연관됨
            'citrus': ['Citrus'],
            'perfume': ['Floral','Herbal'],
            'tropical': ['Tropical'],
            'funky': ['Funky','animal'],
            'herbal': ['Herbal']
        }

    def _build_aroma_wheel_db(self):
        db = {}
        self.flavor_aliases = {} # [추가] 별칭 검색용 딕셔너리 생성
        def add_flavors(category, color, keywords):
            if not keywords: return
            
            # 1. 리스트의 첫 번째 단어를 '대표 단어'로 선정
            primary_key = keywords[0]

            # 2. 대표 단어에만 색상/카테고리 정보 저장
            db[primary_key] = {'category': category, 'color': color}

            for word in keywords:
                self.flavor_aliases[word] = primary_key
        
        add_flavors('forcategory', '#7E6E5C', ['earthy']) # earthy 처리를 위해 추가
        add_flavors('forcategory', '#C9244B', ['fruity'])  # fruity 처리를 위해 추가
        add_flavors('forcategory', '#7E6E5C', ['floral','flower']) 
        add_flavors('forcategory', '#C9244B', ['vegetality'])  
        add_flavors('forcategory', '#7E6E5C', ['woody']) 
        add_flavors('forcategory', '#C9244B', ['malolactic'])
        add_flavors('forcategory', '#C9244B', ['nutty'])
        add_flavors('forcategory', '#C9244B', ['toasty'])
        add_flavors('forcategory', '#C9244B', ['citrus'])
        add_flavors('forcategory', '#C9244B', ['tropical'])
        add_flavors('forcategory', '#C9244B', ['herbal'])
        add_flavors('forcategory', '#C9244B', ['funky'])
        add_flavors('forcategory', '#C9244B', ['red fruit'])
        add_flavors('forcategory', '#C9244B', ['black fruit'])
        add_flavors('forcategory', '#C9244B', ['ripe'])
        add_flavors('forcategory', '#C9244B', ['perfume'])

        # --- FRUITY ---
        add_flavors('Citrus', "#F5EE25", ['lemon'])
        add_flavors('Citrus', '#D6E253', ['lime'])
        add_flavors('Citrus', '#EAD55C', ['grapefruit'])
        add_flavors('Citrus', "#EAB85C", ['tangerine'])
        add_flavors('Citrus', '#F29C33', ['orange peel', 'orange'])
        add_flavors('Pome Fruit', '#D8E289', ['gooseberry'])
        add_flavors('Pome Fruit', '#DCE298', ['pear'])
        add_flavors('Pome Fruit', "#ECD56E", ['apple'])
        add_flavors('Pome Fruit', "#E6C73E", ['quince'])
        add_flavors('Pome Fruit', "#A7D14C", ['green apple'])
        add_flavors('Green Fruit', "#CCE798", ['gooseberry','goose berry'])
        add_flavors('Stone Fruit', '#F7CF6B', ['peach'])
        add_flavors('Stone Fruit', '#F7CF6B', ['apricot'])
        add_flavors('Tropical', '#F4C561', ['melon'])
        add_flavors('Tropical', '#EBB55F', ['guava'])
        add_flavors('Tropical', '#F2D64B', ['pineapple'])
        add_flavors('Tropical', '#E9B949', ['passion fruit', 'passionfruit'])
        add_flavors('Tropical', '#EBC47C', ['lychee'])
        add_flavors('Tropical', '#F2A93B', ['dried apricot'])
        add_flavors('Tropical', "#E9D287", ['banana'])
        add_flavors('Red Berries', "#A81830", ['cherry'])
        add_flavors('Red Berries', '#C9244B', ['currant'])
        add_flavors('Red Berries', '#D93B57', ['raspberry'])
        add_flavors('Red Berries', '#C9244B', ['blackcurrant, cassis'])
        add_flavors('Red Berries', "#BE1940", ['redcurrant'])
        add_flavors('Red Berries', '#BA1E42', ['strawberry'])
        add_flavors('Black Berries', "#571949", ['blackcurrant, cassis'])
        add_flavors('Black Berries', "#52152A", ['blackberry'])
        add_flavors('Black Berries', "#330A14", ['blackcherry'])
        add_flavors('Dried Fruit', "#611E52", ['plum'])
        add_flavors('Dried Fruit', "#2A1536", ['prune'])
        add_flavors('Dried Fruit', "#411111", ['raisin'])

        # --- FLORAL ---
        add_flavors('Floral', "#F7EDC5", ['honeysuckle'])
        add_flavors('Floral', "#DFB4CD", ['hawthorn'])
        add_flavors('Floral', "#F7C4C4", ['orange blossom'])
        add_flavors('Floral', "#D6D39F", ['linden'])
        add_flavors('Floral', "#F7E8F1", ['jasmine'])
        add_flavors('Floral', "#EBE9D1", ['acacia'])
        add_flavors('Floral', "#88316E", ['rose'])
        add_flavors('Floral', "#9B518B", ['lavender'])
        add_flavors('Floral', "#772C81", ['violet'])

        # --- VEGETAL ---
        add_flavors('Vegetal', "#8CB83A", ['capsicum', 'bell pepper'])
        add_flavors('Vegetal', '#96C063', ['fennel'])
        add_flavors('Vegetal', "#B44945", ['rose hip'])
        add_flavors('Vegetal', "#B46945", ['tomato'])
        add_flavors('Vegetal', "#558554", ['cut grass', 'grass'])
        add_flavors('Vegetal', "#6B8836", ['olive'])
        add_flavors('Vegetal', "#389654", ['asparagus'])
        add_flavors('Herbal', '#4E8757', ['cat pee','pee','Boxwood'])
        add_flavors('Herbal', '#4E8757', ['dill'])
        add_flavors('Herbal', '#437B55', ['thyme'])
        add_flavors('Herbal', '#3B7052', ['fern'])
        add_flavors('Herbal', '#34664F', ['mint'])
        add_flavors('Herbal', '#7A823B', ['hay'])
        add_flavors('Herbal', '#606436', ['black tea', 'tea'])
        add_flavors('Herbal', "#806036", ['tobacco'])
        add_flavors('Herbal', '#48633B', ['black currant leaf'])
        add_flavors('Herbal', '#3E5C3C', ['bay leaf'])
        add_flavors('Herbal', '#36523F', ['eucalyptus'])

        # --- MINERAL ---
        add_flavors('Mineral', "#CAD7EB", ['chalk','Limestone'])
        add_flavors('Mineral', "#7E92B1", ['mineral'])
        add_flavors('Mineral', "#7E92B1", ['flint','flinty'])
        add_flavors('Mineral', "#A4B9D8", ['stone', 'wet stone'])
        add_flavors('Mineral', "#8271AA", ['iodine'])
        add_flavors('Mineral', "#738BAC", ['petrol', 'kerosene', 'diesel'])
        add_flavors('Mineral', "#F3E7D0", ['beeswax', 'wax'])

        add_flavors('Earthy', "#683D31", ['mushroom'])
        add_flavors('Earthy', "#5F503E", ['soil', 'dirt'])
        add_flavors('Earthy', "#857257", ['truffle'])
        add_flavors('Earthy', "#5D5F49", ['forest floor'])
        add_flavors('Earthy', "#50616E", ['geosmin'])


        # --- OTHERS ---
        add_flavors('Honey', "#F3C164", ['honey'])
        add_flavors('Honey', "#E7CD9B", ['honeycomb'])
        add_flavors('Honey', "#F3C164", ['marmalade'])
        add_flavors('Yeast', "#CCA26A", ['bread'])
        add_flavors('Malolactic', "#F7F6C6", ['butter','buttery'])
        add_flavors('Malolactic', "#E7E1CD", ['cream'])
        add_flavors('Malolactic', '#EDD9A8', ['yeast'])
        add_flavors('Malolactic', '#EDD9A8', ['milk chocolate'])
        add_flavors('Toasted', "#964A37", ['caramel'])
        add_flavors('Toasted', "#AA6841", ['butterscotch'])
        add_flavors('Toasted', "#8A5D45", ['chocolate', 'cocoa'])
        add_flavors('Toasted', "#6E4D3A", ['toast'])
        add_flavors('Toasted', '#7A5043', ['coffee', 'espresso'])
        add_flavors('Toasted', "#533127", ['mocha'])
        add_flavors('Toasted', '#66423A', ['bacon', 'meaty'])
        add_flavors('Toasted', "#6B3630", ['smoke'])
        add_flavors('Toasted', "#3A211E", ['tar'])
        #add_flavors('Tannin', "#302646", ['tannin'])
        add_flavors('Spice', '#D48642', ['vanilla'])
        add_flavors('Spice', '#CC783B', ['pepper', 'black pepper'])
        add_flavors('Spice', '#C46B35', ['cinnamon'])
        add_flavors('Spice', '#BB5E2F', ['liquorice', 'licorice'])
        add_flavors('Spice', '#B0502A', ['nutmeg'])
        add_flavors('Spice', "#B0972A", ['ginger'])
        add_flavors('Spice', '#A64325', ['clove'])
        add_flavors('Spice', "#69140D", ['anise'])
        add_flavors('Nuts', '#E3A836', ['coconut'])
        add_flavors('Nuts', '#D69830', ['hazelnut'])
        add_flavors('Nuts', '#C9892B', ['almond'])
        add_flavors('Woods', "#64472E", ['oak', 'oaky'])
        add_flavors('Woods', "#815328", ['sandalwood'])
        add_flavors('Woods', '#965725', ['cedar'])
        add_flavors('Woods', "#855E23", ['pine'])
        add_flavors('Woods', "#41332A", ['graphite','lead pencil','pencil shaving'])
        add_flavors('Animal', "#885B40", ['leather', 'saddle'])
        add_flavors('Animal', '#694D47', ['gravy'])
        add_flavors('Animal', "#691B1B", ['game','barnyard'])
        add_flavors('Animal', "#CE865C", ['musk'])
        add_flavors('Sulfuric', "#DFAC4D", ['gun powder'])
        add_flavors('Funky', "#C4A6C5", ['bubble gum','gum'])
        add_flavors('Faults', '#7DC4CC', ['corked', 'musty'])
        add_flavors('Faults', "#502037", ['sherry', 'oxidized']) 
        add_flavors('Faults', '#E84D5B', ['madeira'])
        add_flavors('Faults', "#E97979", ['vinegar'])
        add_flavors('Faults', "#E9BDA4", ['bandaid'])
        add_flavors('Faults', "#D15E81", ['nail polish remover'])
        add_flavors('Faults', '#61A375', ['rubber'])
        add_flavors('Faults', '#89B872', ['onion'])
        add_flavors('Faults', '#4D8076', ['sweet corn'])
        add_flavors('Faults', '#2F5C5A', ['horse sweat'])
        add_flavors('Faults', "#3A0F04", ['brett'])

        return db

    def _get_gaussian(self, x, mu, sigma=0.8, amp=1.0):
        # Numpy 벡터 연산 최적화
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _get_interpolated_color(self, hex_color, factor=0.6):
        # (기존 색상 보간 로직 유지)
        if not hex_color.startswith('#'): return 'white'
        h_val = hex_color.lstrip('#')
        r = int(h_val[0:2], 16); g = int(h_val[2:4], 16); b = int(h_val[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b)
        target_r, target_g, target_b = (255, 255, 255) if lum < 140 else (0, 0, 0)
        new_r = int(r + (target_r - r) * factor)
        new_g = int(g + (target_g - g) * factor)
        new_b = int(b + (target_b - b) * factor)
        return '#{:02x}{:02x}{:02x}'.format(new_r, new_g, new_b)

    # =========================================================================
    # [STEP 1] 데이터 추출 및 저장 (Extract & Save)
    # : raw 리뷰 파일을 읽어 경량화된 json 데이터 파일로 저장합니다.
    # =========================================================================
    def extract_and_save_data(self, input_path, output_path):

        raw_data_storage = {} 
        mention_counts = {}   

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"File Read Error: {e}")
            return False

        for line in lines:
            try:
                record = json.loads(line)
                text = record.get('cleaned_note', '')
                if not text: continue
                
                words = text.split()
                total_words = len(words)
                
                # [핵심 변수 1] 현재 구간의 시작 시간 (기본값 0.0)
                current_base_time = 0.0
                # [핵심 변수 2] 현재 구간이 시작된 단어의 인덱스 (기본값 0)
                current_base_index = 0
                
                found_flavors_in_line = set()

                # -----------------------------------------------------------
                # [핵심 변경] for 문을 while 문으로 변경 (인덱스 점프를 위해)
                # -----------------------------------------------------------
                idx = 0
                while idx < total_words:
                    word = words[idx]
                    word_lower = word.lower().strip('.,!?')
                    
                    matched_key = None
                    step = 1  # 기본적으로 1칸 전진

                    # 1. [Look-ahead] 뒷단어와 합쳐서 DB에 있는지 확인
                    if idx + 1 < total_words:
                        next_word = words[idx+1].lower().strip('.,!?')
                        bigram = f"{word_lower} {next_word}" # 예: "bell pepper"
                        
                        if bigram in self.flavor_aliases:
                            matched_key = self.flavor_aliases[bigram]
                            step = 2  # 두 단어를 사용했으므로 2칸 전진 (pepper 건너뜀)

                    # 2. [Single-word] 두 단어 매칭 실패 시 한 단어만 확인
                    if not matched_key:
                        if word_lower in self.flavor_aliases:
                            matched_key = self.flavor_aliases[word_lower]
                        # 별칭 처리 (복수형 등)
                        elif word_lower.endswith('ies'):
                            singular = word_lower[:-3] + 'y'
                            if singular in self.flavor_aliases: matched_key = self.flavor_aliases[singular]
                        elif word_lower.endswith('s'):
                            singular = word_lower.rstrip('s')
                            if singular in self.flavor_aliases: matched_key = self.flavor_aliases[singular]

                    # 3. 시간축 앵커 감지 (기존 로직 유지)
                    if word_lower in self.section_anchors:
                        new_anchor_time = self.section_anchors[word_lower]
                        if new_anchor_time >= current_base_time:
                            current_base_time = new_anchor_time
                            current_base_index = idx

                    # 4. 데이터 저장 (매칭된 키가 있을 경우)
                    if matched_key:
                        found_flavors_in_line.add(matched_key)
                        
                        # 강도(Amplitude) 계산 (기존 유지)
                        amplitude = 0.5
                        if idx > 0:
                            prev = words[idx-1].lower().strip('.,!?')
                            if prev in self.intensity_modifiers: amplitude *= self.intensity_modifiers[prev]
                            elif idx > 1 and words[idx-2].lower().strip('.,!?') in self.intensity_modifiers:
                                amplitude *= self.intensity_modifiers[words[idx-2].lower().strip('.,!?')]

                        # 위치 매핑 계산 (기존 유지)
                        section_length = max(total_words - current_base_index - 1, 1)
                        relative_index = idx - current_base_index
                        ratio = relative_index / section_length
                        remaining_time_scope = 1.0 - current_base_time
                        pos = current_base_time + (ratio * remaining_time_scope)
                        pos = min(max(pos, 0.0), 1.0)

                        if matched_key not in raw_data_storage:
                            raw_data_storage[matched_key] = {'x': [], 'w': []}
                        
                        raw_data_storage[matched_key]['x'].append(round(pos, 3))
                        raw_data_storage[matched_key]['w'].append(round(amplitude, 2))

                    # 다음 단어로 이동 (1칸 혹은 2칸)
                    idx += step

                # 한 줄 처리가 끝난 후 카운트 집계
                for f_key in found_flavors_in_line:
                    mention_counts[f_key] = mention_counts.get(f_key, 0) + 1

            except json.JSONDecodeError:
                continue
        
        if not mention_counts: return False

        # 2. 노이즈 필터링 (Noise Filtering)
        all_counts = list(mention_counts.values())
        if not all_counts: return False
        
        avg_count = sum(all_counts) / len(all_counts)
        threshold = max(2, avg_count * 0.3) # 평균의 30% 미만 언급은 노이즈 처리
        
        # 필터링된 데이터만 남김
        final_data = {}
        valid_keys = [k for k, v in mention_counts.items() if v >= threshold]
        
        # 3. 부스터 가중치 계산 (Boosting)
        category_multipliers = {}
        for trigger_word, target_categories in self.flavor_families.items():
            count = mention_counts.get(trigger_word, 0)
            if count > avg_count:
                ratio = count / avg_count
                mult = 1.0 + (min(ratio, 2.0) - 1.0) * 0.5 
                for cat in target_categories:
                    category_multipliers[cat] = max(category_multipliers.get(cat, 1.0), mult)

        # 4. 최종 데이터 확정 (가중치 적용하여 저장)
        for key in valid_keys:
            if key not in raw_data_storage: continue
            
            # 부스터 적용 여부 확인
            flavor_info = self.flavor_db.get(key)
            multiplier = 1.0
            if flavor_info:
                cat = flavor_info.get('category')
                multiplier = category_multipliers.get(cat, 1.0)
            
            # 가중치 리스트 전체에 곱하기 (numpy 쓰면 빠르지만, 저장 단계라 list comprehension 사용)
            weights = raw_data_storage[key]['w']
            if multiplier > 1.0:
                weights = [round(w * multiplier, 2) for w in weights]
            
            # 데이터 구조화: 요청하신대로 x와 w 리스트 저장
            final_data[key] = {
                'x': raw_data_storage[key]['x'],
                'w': weights,
                'count': mention_counts[key] # 나중에 랭킹 산정용으로 저장
            }
            
        # 5. 파일로 저장
        try:
            # (1) 먼저 기본 포맷으로 JSON 문자열 생성
            json_str = json.dumps(final_data, indent=4, ensure_ascii=False)

            # (2) 정규표현식을 사용하여 숫자 배열 부분만 한 줄로 압축
            # 패턴 설명: [ (공백/줄바꿈) 숫자,점,쉼표,마이너스 등 (공백/줄바꿈) ] 형태를 찾음
            json_str = re.sub(
                r'\[\s+([\d\.\,\s\-]+?)\s+\]', 
                lambda m: '[' + ' '.join(m.group(1).split()) + ']', 
                json_str
            )

            # (3) 파일 쓰기
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
                
            print(f"Analyzed data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False

    # =========================================================================
    # [STEP 2] 데이터 로드 및 시각화 (Load & Render)
    # : 저장된 json 데이터를 불러와 매우 빠르게 그래프를 그립니다.
    # =========================================================================
    def create_graph_from_data(self, data_file_path):
        
        """
        저장된 분석 데이터(.json)를 읽어서 Streamgraph Figure를 반환합니다.
        NLP 분석 과정이 생략되므로 속도가 매우 빠릅니다.
        """
        # 1. 데이터 로드
        if not os.path.exists(data_file_path):
            return Figure() # 빈 피규어 반환
            
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                flavor_data = json.load(f)
        except:
            return Figure()

        if not flavor_data:
            return Figure()

        # 2. 곡선 생성 (Curve Generation)
        # 이제 저장된 포인트들을 기반으로 KDE(Kernel Density Estimation) 느낌의 곡선을 만듭니다.
        x_axis = np.linspace(-0.2, 1.2, 600)
        aggregated_curves = {}
        
        # 랭킹 산정을 위해 최대 언급 횟수 파악
        max_mention = 0
        for fname, data in flavor_data.items():
             # count가 저장되어 있으면 쓰고, 없으면 리스트 길이로 추정
            cnt = data.get('count', len(data['x']))
            if cnt > max_mention: max_mention = cnt


        for flavor, data in flavor_data.items():
            if flavor in self.flavor_families.keys(): continue # "Fruity" 같은 추상적 키워드는 제외
            
            positions = data['x']
            weights = data['w']
            count = data.get('count', len(positions))
            
            if not positions: continue

            # [최적화] 모든 포인트에 대해 가우시안을 더합니다.
            # 포인트가 너무 많으면(예: 1000개 이상) 다운샘플링 할 수도 있으나,
            # Numpy 벡터 연산은 수천 개 정도는 순식간입니다.
            curve = np.zeros_like(x_axis)
            
            # 1. 브로드캐스팅을 위해 차원 늘리기 (배열 -> 컬럼 벡터)
            # 형태 변환: [0.1, 0.5] -> [[0.1], [0.5]] (N행 1열)
            mu_vector = np.array(positions)[:, np.newaxis]
            amp_vector = np.array(weights)[:, np.newaxis]
            
            # 2. 함수 호출
            # x_axis는 (300,)이고 mu는 (N, 1)이므로, 결과는 자동으로 (N, 300)이 됩니다.
            sigma_val = 0.1 # 점들이 서로 잘 뭉치도록 설정
            
            gaussians = self._get_gaussian(x_axis, mu=mu_vector, sigma=sigma_val, amp=amp_vector)
            
            # 3. 합치기 (Curve 생성)
            curve = np.sum(gaussians, axis=0)

            # [Sculpting] 모양 다듬기 (기존 로직 계승)
            # 1. 랭킹 가중치
            ratio = count / max_mention if max_mention > 0 else 0
            rank_weight = np.interp(ratio, 
                        [0.0, 0.03, 0.08, 0.1, 0.33, 0.5, 1.0], 
                        [0.0, 0.0, 0.3, 0.4, 0.6, 0.7, 1.0])
            
            # 2. 정규화 및 샤프닝
            peak_height = np.max(curve)
            if peak_height > 0.1: # 너무 작은 노이즈 제거
                curve = curve / peak_height # 0~1로 정규화
                curve = np.power(curve, 3)  # 뚱뚱한 곡선을 날렵하게(Cubed)
                curve = curve * rank_weight # 빈도수에 따른 최종 높이 조절
                
                aggregated_curves[flavor] = curve

        # 3. 정렬 및 필터링 (Sorting & Grouping) - 기존 로직과 동일
        all_sorted = sorted(aggregated_curves.items(), key=lambda item: np.sum(item[1]), reverse=True)
        
        final_candidates = []
        category_counts = {} 
        MAX_TOTAL = 30      
        MAX_PER_CAT = 4      

        for flavor, curve in all_sorted:
            if len(final_candidates) >= MAX_TOTAL: break
            
            # DB에 없는 맛이 들어왔을 경우 안전처리
            f_info = self.flavor_db.get(flavor)
            if not f_info: continue 

            cat = f_info.get('category', 'Unknown')
            current_count = category_counts.get(cat, 0)
            
            if current_count < MAX_PER_CAT:
                final_candidates.append((flavor, curve))
                category_counts[cat] = current_count + 1

        if not final_candidates:
            return Figure()

        # 4. 카테고리별 그룹화 (Grouping for Blur)
        grouped_dict = {}
        for flav, curve in final_candidates:
            cat = self.flavor_db.get(flav).get('category')
            if cat not in grouped_dict: grouped_dict[cat] = []
            grouped_dict[cat].append((flav, curve))
            
        sorted_groups = []
        # 그룹 총량 순 정렬
        group_keys = sorted(grouped_dict.keys(), 
                          key=lambda c: sum([np.sum(x[1]) for x in grouped_dict[c]]), 
                          reverse=True)
                          
        for cat in group_keys:
            items = grouped_dict[cat]
            # 그룹 내부는 'Nose(초반)' 강도 순 정렬
            nose_idx = int(300 * 0.15)
            items.sort(key=lambda item: item[1][nose_idx], reverse=True)
            sorted_groups.extend(items)

        # 5. 그래프 그리기 (Rendering) - Matplotlib
        fig = Figure(figsize=(16, 8), dpi=100, facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        ax.set_facecolor=('#1e1e1e')

        y_stack_list = []
        labels = []
        colors = []
        categories = []

        for flavor, y_values in sorted_groups:
            y_stack_list.append(y_values)
            labels.append(flavor.upper())
            info = self.flavor_db.get(flavor)
            colors.append(info['color'])
            categories.append(info['category'])

        # --- [추가] 카테고리별 블러 제어 마스크 계산 ---
        # 카테고리 내 모든 flavor의 수치를 곱하여, 하나라도 0이면 블러가 0이 되도록 함
        category_blur_masks = {}
        unique_cats = set(categories)
        for c_name in unique_cats:
            # 해당 카테고리에 속한 모든 y_values 리스트 추출
            cat_y_list = [y_stack_list[j] for j, cn in enumerate(categories) if cn == c_name]
            
            if len(cat_y_list) > 1:
                # 모든 요소를 곱함 (어느 하나가 0이면 결과는 0)
                prod = np.prod(cat_y_list, axis=0)
                # 정규화 (0~1 사이로 변환하여 블러 강도 계수로 사용)
                max_p = np.max(prod)
                if max_p > 1e-9:
                    category_blur_masks[c_name] = prod / max_p
                else:
                    category_blur_masks[c_name] = np.zeros_like(x_axis)
            else:
                # flavor가 하나뿐인 카테고리는 자기 자신의 두께에 비례하도록 설정
                single_y = cat_y_list[0]
                max_y = np.max(single_y)
                category_blur_masks[c_name] = (single_y / max_y) if max_y > 0 else np.zeros_like(x_axis)

        # --- Stack Drawing Logic (기존 로직 수정) ---
        total_y = np.sum(y_stack_list, axis=0)
        current_bottom = -0.5 * total_y 
        num_layers = len(y_stack_list)

        for i in range(num_layers):
            y = y_stack_list[i]
            color = colors[i]
            cat = categories[i]
            
            # 현재 카테고리의 마스크 가져오기
            blur_mask = category_blur_masks.get(cat, np.ones_like(x_axis))
            
            center = current_bottom + (y / 2)
            radius = y / 2
            
            # Blur Effects
            prev_cat = categories[i-1] if i > 0 else None
            next_cat = categories[i+1] if i < num_layers - 1 else None
            
            blur_factors = [2.0, 1.9, 1.8, 1.4, 1.2] 
            blur_alphas  = [0.1, 0.2, 0.25, 0.3, 0.36]

            for factor, alpha in zip(blur_factors, blur_alphas):
                # 기존 scale 값에 blur_mask를 곱해 적용
                # factor가 1.0보다 큰 부분(확장분)에 대해서만 마스크를 적용하여 
                # 두께가 얇아지는 곳에서 블러가 수축되도록 함
                effective_scale_up = 1.0 + (factor - 1.0) * blur_mask if (cat == next_cat) else 1.0
                effective_scale_down = 1.0 + (factor - 1.0) * blur_mask if (cat == prev_cat) else 1.0
                
                y1 = center - (radius * effective_scale_down)
                y2 = center + (radius * effective_scale_up)
                ax.fill_between(x_axis, y1, y2, color=color, alpha=alpha, linewidth=0)

            # Main Body (실제 데이터 곡선)
            ax.fill_between(x_axis, current_bottom, current_bottom + y, color=color, alpha=0.9, linewidth=0)
            current_bottom += y

        # Masking & Labels (기존 코드의 라벨링 및 마스킹 로직 통합)
        self._apply_masking_and_labels(ax, x_axis, y_stack_list, labels, colors, total_y)
        
        fig.tight_layout(pad=0)
        return fig

    def create_mini_graph(self, data_file_path):
        """
        [AnalyticsTab 전용] 
        여백을 완전히 제거하고 데이터 크기에 맞춰 꽉 차게 그리는 미니 그래프
        """
        if not os.path.exists(data_file_path): return None
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f: flavor_data = json.load(f)
        except: return None
        if not flavor_data: return None

        # 1. 곡선 데이터 생성
        x_axis = np.linspace(-0.2, 1.2, 100) 
        aggregated_curves = {}
        max_mention = 0
        for fname, data in flavor_data.items():
            cnt = data.get('count', len(data['x']))
            if cnt > max_mention: max_mention = cnt

        for flavor, data in flavor_data.items():
            if flavor in self.flavor_families.keys(): continue 
            positions = data['x']
            weights = data['w']
            count = data.get('count', len(positions))
            if not positions: continue

            curve = np.zeros_like(x_axis)
            mu_vector = np.array(positions)[:, np.newaxis]
            amp_vector = np.array(weights)[:, np.newaxis]
            sigma_val = 0.1 
            gaussians = self._get_gaussian(x_axis, mu=mu_vector, sigma=sigma_val, amp=amp_vector)
            curve = np.sum(gaussians, axis=0)

            ratio = count / max_mention if max_mention > 0 else 0
            rank_weight = np.interp(ratio, [0.0, 0.33, 0.5, 1.0], [0.4, 0.5, 0.8, 1.0])
            peak_height = np.max(curve)
            if peak_height > 0.1: 
                curve = curve / peak_height 
                curve = np.power(curve, 3)  
                curve = curve * rank_weight 
                aggregated_curves[flavor] = curve

        # 2. 정렬 및 그룹화
        all_sorted = sorted(aggregated_curves.items(), key=lambda item: np.sum(item[1]), reverse=True)
        final_candidates = []
        category_counts = {} 
        MAX_TOTAL = 30
        MAX_PER_CAT = 4      

        for flavor, curve in all_sorted:
            if len(final_candidates) >= MAX_TOTAL: break
            f_info = self.flavor_db.get(flavor)
            if not f_info: continue 
            cat = f_info.get('category', 'Unknown')
            current_count = category_counts.get(cat, 0)
            if current_count < MAX_PER_CAT:
                final_candidates.append((flavor, curve))
                category_counts[cat] = current_count + 1

        if not final_candidates: return None

        grouped_dict = {}
        for flav, curve in final_candidates:
            cat = self.flavor_db.get(flav).get('category')
            if cat not in grouped_dict: grouped_dict[cat] = []
            grouped_dict[cat].append((flav, curve))
            
        sorted_groups = []
        group_keys = sorted(grouped_dict.keys(), key=lambda c: sum([np.sum(x[1]) for x in grouped_dict[c]]), reverse=True)
        for cat in group_keys:
            items = grouped_dict[cat]
            nose_pos = 0.15
            idx_ratio = (nose_pos - (-0.2)) / (1.2 - (-0.2))
            nose_idx = int(len(x_axis) * idx_ratio)
            items.sort(key=lambda item: item[1][nose_idx], reverse=True)
            sorted_groups.extend(items)

        # 3. 그래프 설정 (여백 제거)
        BG_COLOR = '#333333' 
        # figsize를 조금 더 키워 캔버스 자체를 확보
        fig = Figure(figsize=(3.4, 1.8), dpi=30, facecolor=BG_COLOR)
        
        # [핵심 1] 서브플롯 여백 완전 제거 (왼쪽/오른쪽/위 딱 붙이고, 아래는 텍스트 공간 조금 남김)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_COLOR)
        ax.axis('off')

        y_stack_list = []
        colors = []
        for flavor, y_values in sorted_groups:
            y_stack_list.append(y_values)
            info = self.flavor_db.get(flavor)
            colors.append(info['color'])

        # 4. 스트림 그래프 그리기
        total_y = np.sum(y_stack_list, axis=0)
        current_bottom = -0.5 * total_y 

        for i in range(len(y_stack_list)):
            y = y_stack_list[i]
            color = colors[i]
            ax.fill_between(x_axis, current_bottom, current_bottom + y, color=color, alpha=0.9, linewidth=0)
            current_bottom += y

        # 5. [핵심 2] 동적 Y축 스케일링 (꽉 차게 만들기)
        if len(total_y) > 0:
            max_h = np.max(total_y)
            # 최대 높이의 55%만 위아래 한계로 잡음 (꽉 차게 줌인)
            # 기존 0.75 -> 0.55로 줄여서 그래프를 위아래로 늘림
            limit = max_h * 0.55
            ax.set_ylim(-limit, limit)
        else:
            ax.set_ylim(-1.0, 1.0)

        # 6. 구분선 및 라벨
        # (1) 구분선
        ax.axvline(x=0.3, color='white', linestyle=':', alpha=0.2, linewidth=0.8, zorder=0)
        ax.axvline(x=0.7, color='white', linestyle=':', alpha=0.2, linewidth=0.8, zorder=0)

        # (2) 라벨 텍스트 (transform=ax.transAxes 사용 -> 데이터 크기 상관없이 위치 고정)
        font_style = {'color': "#aaaaaa", 'fontsize': 8, 'fontweight': 'bold', 'ha': 'center', 'va': 'bottom'}
        
        # transform=ax.transAxes: (0,0)이 왼쪽 아래, (1,1)이 오른쪽 위
        # y=0.02: 바닥에서 아주 살짝 띄움
        # x값은 데이터 좌표가 아니라 0.0~1.0 비율이므로 변환 필요
        # 데이터 범위 -0.2 ~ 1.2 (총 1.4)
        # 0.15 위치 비율: (0.15 - (-0.2)) / 1.4 = 0.25
        # 0.50 위치 비율: (0.50 - (-0.2)) / 1.4 = 0.50
        # 0.85 위치 비율: (0.85 - (-0.2)) / 1.4 = 0.75
        
        ax.set_xlim(-0.1, 1.1)
        
        # tight_layout 호출 안함 (subplots_adjust로 수동 제어했으므로)
        return fig

    def _apply_masking_and_labels(self, ax, x_axis, y_stack_list, labels, colors, total_y):
        # 코드가 길어져서 분리한 마스킹 및 라벨링 헬퍼 함수
        # (기존 create_aggregate_streamgraph 하단의 로직을 그대로 사용)
        
        # 1. Background Masking
        graph_top_boundary = total_y / 2.0
        graph_bottom_boundary = -total_y / 2.0
        mask_limit = np.max(total_y) * 2.0
        bg_color = '#1e1e1e'
        
        ax.fill_between(x_axis, graph_top_boundary, mask_limit, color=bg_color, linewidth=0, zorder=3)
        ax.fill_between(x_axis, -mask_limit, graph_bottom_boundary, color=bg_color, linewidth=0, zorder=3)
        ax.set_xlim(-0.05, 1.05)

        # Y축 범위 미리 계산
        y_visual_max = np.max(total_y) * 0.6
        y_offset = np.max(total_y) * 0.20
        y_top_limit = (y_visual_max + y_offset) * (1.0 - 0.33)
        label_margin = np.max(total_y) * 0.10
        min_graph_bottom = np.min(graph_bottom_boundary)
        y_bottom_limit = (min_graph_bottom - label_margin) - np.max(total_y) * 0.05

        # [중요] 라벨을 그리기 전에 미리 축 범위를 확정지어야 함
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(y_bottom_limit, y_top_limit)
        ax.axis('off')

        # 2. Text Labeling
        current_bottom_array = -0.5 * total_y
        min_thickness_threshold = np.max(total_y) * 0.02

        for i, y_values in enumerate(y_stack_list):
            flavor_name = labels[i]
            bg_color_hex = colors[i]
            
            peak_idx = np.argmax(y_values)
            center_line_array = current_bottom_array + (y_values / 2)
            center_y = center_line_array[peak_idx]
            peak_height = y_values[peak_idx]
            current_bottom_array += y_values

            if peak_height < min_thickness_threshold: continue

            # --- 픽셀 기반 각도 계산 ---
            step = 10 # 유칼립투스 같은 급경사를 잡기 위해 좁은 범위 관찰
            idx_prev = max(0, peak_idx - step)
            idx_next = min(len(x_axis) - 1, peak_idx + step)

            # 데이터 좌표를 픽셀 좌표로 변환 (ax.transData 사용)
            p_prev = ax.transData.transform((x_axis[idx_prev], center_line_array[idx_prev]))
            p_next = ax.transData.transform((x_axis[idx_next], center_line_array[idx_next]))

            # 픽셀 변위로 각도 계산 (보정 계수 0.7 제거, 1.0 사용)
            d_x = p_next[0] - p_prev[0]
            d_y = p_next[1] - p_prev[1]
            rotation_angle = math.degrees(math.atan2(d_y, d_x))

            # 가독성 한계 각도 완화
            MAX_ANGLE = 55
            rotation_angle = max(-MAX_ANGLE, min(MAX_ANGLE, rotation_angle))

            final_fontsize = max(8, min(20, int(9 + (peak_height * peak_height * 11))))
            text_color = self._get_interpolated_color(bg_color_hex, factor=0.6)
            
            ax.text(x_axis[peak_idx], center_y, flavor_name,
                    ha='center', va='center', fontsize=final_fontsize, fontweight='bold',
                    color=text_color, rotation=rotation_angle, rotation_mode='anchor', zorder=10)

        # 3. Axis & Limits
        y_visual_max = np.max(total_y) * 0.6
        y_offset = np.max(total_y) * 0.20
        y_top_limit = (y_visual_max + y_offset) * (1.0 - 0.33)
        
        label_margin = np.max(total_y) * 0.10
        min_graph_bottom = np.min(graph_bottom_boundary)
        y_bottom_limit = (min_graph_bottom - label_margin) - np.max(total_y) * 0.05
        
        ax.set_ylim(y_bottom_limit, y_top_limit)
        ax.axis('off')

        # 4. Custom Axis Labels
        label_y_pos = -y_visual_max * 1.02
        section_style = {'color': "#837E7E", 'fontsize': 12, 'fontweight': 'bold', 'ha': 'center', 'va': 'bottom', 'zorder': 20}
        ax.text(0.15, label_y_pos, 'NOSE', **section_style)
        ax.text(0.50, label_y_pos, 'PALATE', **section_style)
        ax.text(0.85, label_y_pos, 'FINISH', **section_style)

        ax.axvline(x=0.3, color='white', linestyle=':', alpha=0.1, zorder=5)
        ax.axvline(x=0.7, color='white', linestyle=':', alpha=0.1, zorder=5)

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # [수정] 배경색을 메인 배경(#1e1e1e)과 일치시켜 빈 공간이 튀지 않게 함
        bg_color = '#1e1e1e' 
        
        self.canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # 내부 프레임도 배경색 일치
        self.scrollable_frame = tk.Frame(self.canvas, bg=bg_color)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 5)) # 캔버스 오른쪽 여백
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class SearchTab(ttk.Frame):
    def __init__(self, parent, metadata, analyzer):
        super().__init__(parent)
        self.metadata = metadata 
        self.analyzer = analyzer
        self.current_results = [] # 현재 검색 결과 저장용 (정렬을 위해)
        
        # 메인 컨테이너
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill='both', expand=True)

        # 1. 초기 중앙 검색 화면 (Initial View)
        self.initial_view = tk.Frame(self.main_container, bg='#1e1e1e')
        self._init_initial_view()

        # 2. 결과 리스트 화면 (Results View)
        self.results_view = tk.Frame(self.main_container, bg='#1e1e1e')
        self._init_results_view()

        # 3. 상세 화면 (Detail View)
        self.detail_view = tk.Frame(self.main_container, bg='#1e1e1e')
        self._init_detail_view()

        # 시작은 초기 화면
        self.show_initial_view()

    # --- 화면 초기화 ---
    def _init_initial_view(self):
        # 중앙 배치 컨테이너
        center_frame = tk.Frame(self.initial_view, bg='#1e1e1e')
        center_frame.place(relx=0.5, rely=0.4, anchor='center')

        lbl_logo = tk.Label(center_frame, text="Search Wine", font=('Helvetica', 32, 'bold'), bg='#1e1e1e', fg='white')
        lbl_logo.pack(pady=(0, 30))

        self.init_search_var = tk.StringVar()
        
        # [수정] 검색창 디자인: Frame으로 감싸서 내부 여백(Padding) 만들기
        # 1. 겉을 감싸는 박스 (배경색 역할)
        search_box = tk.Frame(center_frame, bg='#333333', height=50)
        search_box.pack(ipady=2) # 박스 높이 확보

        # 2. 실제 입력창 (테두리 없음, 부모색과 통일)
        entry = tk.Entry(search_box, textvariable=self.init_search_var, 
                         font=('Arial', 18), width=35, # 너비는 여기서 조절
                         bg='#333333', 
                         fg='white', 
                         insertbackground='white', 
                         relief='flat')
        
        # 3. 입력창을 박스 안에 넣을 때 padx로 왼쪽 여백 확보!
        entry.pack(fill='both', expand=True, padx=15, pady=8)
        
        entry.bind("<Return>", lambda e: self.perform_search(query_source='initial'))
        entry.focus()

    def _init_results_view(self):
        # 상단 바
        top_bar = tk.Frame(self.results_view, bg='#1e1e1e', pady=20)
        top_bar.pack(side='top', fill='x')

        self.res_search_var = tk.StringVar()
        
        # [수정] Frame Wrapper 방식으로 여백 확보
        # 1. 검색 박스 컨테이너
        search_box = tk.Frame(top_bar, bg='#333333')
        search_box.pack(side='left', padx=(20, 10), ipady=1) # 위치 잡기

        # 2. 입력창
        entry = tk.Entry(search_box, textvariable=self.res_search_var, 
                         font=('Arial', 14), width=30,
                         bg='#333333', 
                         fg='white', 
                         insertbackground='white',
                         relief='flat')
                         
        # 3. 내부 여백(padx=10) 적용
        entry.pack(side='left', fill='both', expand=True, padx=10, pady=5)
        entry.bind("<Return>", lambda e: self.perform_search(query_source='results'))

        # 검색 버튼
        btn_search = tk.Button(top_bar, text="🔍", font=('Arial', 12), 
                               bg='#333333', fg='white', 
                               activebackground='#555555', activeforeground='white',
                               relief='flat', cursor='hand2',
                               command=lambda: self.perform_search(query_source='results'))
        btn_search.pack(side='left', ipady=1)

        # 정렬 콤보박스 (기존 코드 유지)
        self.sort_var = tk.StringVar(value="Rating")
        sort_combo = ttk.Combobox(top_bar, textvariable=self.sort_var, state="readonly", font=('Arial', 11), width=10)
        sort_combo['values'] = ("Rating", "A-Z")
        sort_combo.pack(side='right', padx=20) 
        sort_combo.bind("<<ComboboxSelected>>", self.sort_results)
        
        lbl_sort = tk.Label(top_bar, text="Sort by:", bg='#1e1e1e', fg='#aaaaaa', font=('Arial', 11))
        lbl_sort.pack(side='right', padx=(0, 10))

        # 리스트 영역 (기존 코드 유지)
        self.result_area = ScrollableFrame(self.results_view)
        self.result_area.pack(side='bottom', fill='both', expand=True, padx=(20, 10), pady=(0, 10))

    def _init_detail_view(self):
        # 상단 네비게이션
        nav_frame = tk.Frame(self.detail_view, bg='#1e1e1e', pady=10)
        nav_frame.pack(side='top', fill='x')
        
        btn_back = ttk.Button(nav_frame, text="⬅", command=self.back_to_results)
        btn_back.pack(side='left', padx=20)

        self.lbl_detail_title = tk.Label(nav_frame, text="", font=('Helvetica', 18, 'bold'), bg='#1e1e1e', fg='white')
        self.lbl_detail_title.pack(side='left', padx=20)

        # 그래프 영역
        self.graph_frame = tk.Frame(self.detail_view, bg='#1e1e1e')
        self.graph_frame.pack(side='bottom', fill='both', expand=True)

    # --- 화면 전환 ---
    def show_initial_view(self):
        self.results_view.pack_forget()
        self.detail_view.pack_forget()
        self.initial_view.pack(fill='both', expand=True)

    def show_results_view(self):
        self.initial_view.pack_forget()
        self.detail_view.pack_forget()
        self.results_view.pack(fill='both', expand=True)

    def show_detail_view(self, wine_data):
        self.initial_view.pack_forget()
        self.results_view.pack_forget()
        self.detail_view.pack(fill='both', expand=True)
        
        # 타이틀 설정
        #region = wine_data.get('region', '')
        #country = wine_data.get('country', '')
        title = f"{wine_data.get('name', 'Unknown')}"
        #if region: title += f"  ({region})"
        self.lbl_detail_title.config(text=title)
        
        self._draw_graph(wine_data)

    def back_to_results(self):
        # 뒤로가기 시 리스트 화면으로 복귀
        self.show_results_view()

    # --- 기능 로직 ---
    def _normalize_text(self, text):
        """
        독일어 움라우트(ö, ü) 및 프랑스어 악센트를 
        영어 알파벳(o, u, a)으로 안전하게 변환하는 정규화 로직
        """
        if text is None: return ""
        text = str(text)
        
        # 1. 유니코드 분해 (NFD: 'ö'를 'o'와 '¨'로 나눔)
        nfd_text = unicodedata.normalize('NFD', text)
        
        # 2. 'Mn'(Mark, Nonspacing) 카테고리(악센트 기호)만 필터링하고 다시 합침
        # 이렇게 하면 'o'는 남고 위쪽의 점 두개(¨)만 사라집니다.
        clean_text = "".join([c for c in nfd_text if unicodedata.category(c) != 'Mn'])
        
        # 3. 소문자 변환 및 공백 제거
        return clean_text.lower().strip()

    def perform_search(self, query_source='results'):
        # 1. 검색어 가져오기
        if query_source == 'initial':
            raw_query = self.init_search_var.get()
            self.res_search_var.set(raw_query) 
        else:
            raw_query = self.res_search_var.get()

        # 검색어가 없으면 리턴
        if not raw_query: return

        # 2. 검색어 정규화
        query = self._normalize_text(raw_query)
        
        # [디버깅] 콘솔에 현재 상태 출력 (문제가 뭔지 바로 알 수 있음)
        #print(f"🔎 [Search Debug] Raw: '{raw_query}' -> Normalized: '{query}'")
        #print(f"📊 [Data Debug] Total Metadata Count: {len(self.metadata)}")

        self.current_results = []
        
        # 3. 데이터 순회하며 검색
        for w in self.metadata:
            # 데이터 가져오기 (없으면 빈 문자열)
            name_raw = w.get('name', '')
            winery_raw = w.get('winery', '')
            region_raw = w.get('region', '')
            
            # 정규화 (대소문자, 악센트 제거)
            name = self._normalize_text(name_raw)
            winery = self._normalize_text(winery_raw)
            
            # 지역은 리스트일 수도 있고 문자열일 수도 있음
            if isinstance(region_raw, list):
                region = self._normalize_text(" ".join(map(str, region_raw)))
            else:
                region = self._normalize_text(region_raw)
            
            # [핵심] 부분 일치 검사 (in 연산자)
            # "tal" in "chateau talbot" -> True가 되어야 함
            if (query in name) or (query in winery) or (query in region):
                self.current_results.append(w)
                
        # [디버깅] 검색된 개수 출력
        #print(f"✅ [Result Debug] Found: {len(self.current_results)} wines")

        # 4. 화면 갱신
        self.show_results_view()
        self.update_result_list()

    def sort_results(self, event=None):
        # 정렬 옵션 변경 시 호출
        self.update_result_list()

    def update_result_list(self):
        # 1. 기존 리스트 클리어
        for widget in self.result_area.scrollable_frame.winfo_children():
            widget.destroy()

        if not self.current_results:
            lbl_none = tk.Label(self.result_area.scrollable_frame, text="No wines found.", bg='#1e1e1e', fg='gray', font=('Arial', 14))
            lbl_none.pack(pady=50)
            return

        # 2. 정렬 실행
        sort_mode = self.sort_var.get()
        if sort_mode == "Rating":
            # 평점 높은 순 (내림차순)
            # rating이 없는 경우 0.0 처리
            self.current_results.sort(key=lambda x: float(x.get('rating', 0) or 0), reverse=True)
        elif sort_mode == "A-Z":
            # 이름 순 (오름차순)
            self.current_results.sort(key=lambda x: x.get('name', '').lower())

        # 3. 카드 생성 (최대 50개 제한)
        for wine in self.current_results[:50]:
            self.create_wine_card(wine)

    def create_wine_card(self, wine):
        
        # ---------------------------------------------------------
        # 1. [핵심 수정] 실제 리뷰 파일 라인 수 카운팅 (캐싱 적용)
        # ---------------------------------------------------------
        wine_id = wine.get('id')
        
        # (1) 이미 세어본 적이 있는지 확인 (메모리 캐싱) -> 스크롤 버벅임 방지
        if 'cached_review_count' in wine:
            review_count = wine['cached_review_count']
        else:
            # (2) 없다면 파일 직접 열어서 카운팅
            file_path = os.path.join("cleaned", f"wine_{wine_id}_clean.jsonl")
            if os.path.exists(file_path):
                try:
                    # 라인 수 세기 (제너레이터를 사용하여 메모리 효율적)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_count = sum(1 for _ in f)
                except:
                    review_count = 0
            else:
                review_count = 0
            
            # (3) 결과 저장 (다음번엔 파일 안 열도록)
            wine['cached_review_count'] = review_count
            
        # [조건] 리뷰 수가 3개 이하면 비활성화 (기준은 3, 5, 20 등 원하시는 대로 수정 가능)
        # 파일이 아예 없거나(0) 너무 적으면 분석 불가하므로 비활성화
        is_disabled = review_count <= 3

        # 2. 스타일 설정
        if is_disabled:
            BG_NORMAL = '#222222'     
            BG_HOVER = '#222222'      
            FG_PRIMARY = '#555555'    
            FG_SECONDARY = '#444444'  
            CURSOR = 'arrow'          
            image_brightness = 0.3    
            BOTTOM_TEXT_COLOR = '#333333' # 비활성화 시 아주 어둡게
        else:
            BG_NORMAL = '#333333'     
            BG_HOVER = '#3e3e3e'      
            FG_PRIMARY = 'white'      
            FG_SECONDARY = '#aaaaaa'  
            CURSOR = 'hand2'          
            image_brightness = 1.0    
            BOTTOM_TEXT_COLOR = '#777777' # 평소 색상 (어두운 회색)
            BOTTOM_TEXT_HOVER = '#bbbbbb' # [NEW] 호버 시 밝은 회색

        # =========================================================
        # [설정] 카드 크기 및 레이아웃 고정값
        # =========================================================
        CARD_HEIGHT = 300     # 카드 높이
        IMAGE_BOX_WIDTH = 300 # [핵심] 이미지 영역의 고정 너비 (글자는 이 뒤에서 시작)

        # 1. 카드 프레임
        card = tk.Frame(self.result_area.scrollable_frame, bg=BG_NORMAL, bd=0, height=CARD_HEIGHT, cursor=CURSOR)
        # ipady=0, pady=8 (카드 간 간격)
        card.pack(fill='x', pady=8, padx=5, ipady=0) 
        
        # 레이아웃이 뭉개지지 않도록 프레임 크기 고정 (높이 220 유지)
        card.pack_propagate(False)

        def on_click(e):
            if not is_disabled:
                self.show_detail_view(wine)
            else:
                # (옵션) 비활성화 카드 클릭 시 안내 메시지를 띄우고 싶다면 주석 해제
                # messagebox.showinfo("Info", f"Not enough reviews to analyze (Found: {review_count}).")
                pass

        # ---------------------------------------------------------
        # 2. [핵심 변경] 이미지 컨테이너 (너비 고정 박스)
        # ---------------------------------------------------------
        # 이 프레임은 무조건 너비 220px, 높이 220px를 차지합니다.
        img_container = tk.Frame(card, bg=BG_NORMAL, width=IMAGE_BOX_WIDTH, height=CARD_HEIGHT)
        img_container.pack(side='left', fill='y')
        img_container.pack_propagate(False) # 내용물 크기에 따라 줄어들지 않게 고정

        # 이미지 로드
        has_image = wine.get('image', 0)
        image_path = resource_path(os.path.join("image", f"wine_{wine_id}_image.png"))
        img_widget = None

        if has_image == 1 and os.path.exists(image_path):
            try:
                from PIL import ImageEnhance # 밝기 조절을 위해 추가 임포트 필요할 수 있음

                pil_img = Image.open(image_path).convert("RGBA")
                orig_w, orig_h = pil_img.size

                # 크롭 및 리사이즈 (기존 로직)
                TOP_CROP_RATIO = 0.40 
                BOTTOM_CROP_RATIO = 0.05 
                if orig_h > 50: 
                    top_cut = int(orig_h * TOP_CROP_RATIO)
                    bottom_cut = int(orig_h * (1 - BOTTOM_CROP_RATIO))
                    cropped_img = pil_img.crop((0, top_cut, orig_w, bottom_cut))
                else:
                    cropped_img = pil_img

                crop_w, crop_h = cropped_img.size
                aspect_ratio = crop_w / crop_h
                new_height = CARD_HEIGHT
                new_width = int(new_height * aspect_ratio)
                
                resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 배경 합성
                # 비활성화일 경우 배경색도 어둡게 맞춰줌
                r, g, b = self.winfo_rgb(BG_NORMAL)
                bg_color_tuple = (r//256, g//256, b//256, 255)
                
                background = Image.new('RGBA', resized_img.size, bg_color_tuple) 
                combined_img = Image.alpha_composite(background, resized_img)
                final_img = combined_img.convert("RGB").filter(ImageFilter.SHARPEN)

                # [핵심] 비활성화 시 이미지 어둡게 만들기
                if is_disabled:
                    enhancer = ImageEnhance.Brightness(final_img)
                    final_img = enhancer.enhance(image_brightness) # 0.3배 밝기

                tk_img = ImageTk.PhotoImage(final_img)
                
                img_widget = tk.Label(img_container, image=tk_img, bg=BG_NORMAL, bd=0)
                img_widget.image = tk_img 
            except Exception as e:
                # print(f"이미지 로드 실패: {e}")
                img_widget = None

        if img_widget is None:
            # 이미지 없을 때 플레이스홀더
            placeholder_img = Image.new('RGB', (80, CARD_HEIGHT), color='#222222' if is_disabled else '#444444')
            tk_placeholder = ImageTk.PhotoImage(placeholder_img)
            img_widget = tk.Label(img_container, image=tk_placeholder, text="No\nImg", 
                                  font=('Arial', 12), fg=FG_PRIMARY,
                                  compound='center', bg=BG_NORMAL, bd=0)
            img_widget.image = tk_placeholder 

        if img_widget:
            img_widget.place(relx=0.5, rely=0.5, anchor='center')

        # ---------------------------------------------------------
        # 4. 텍스트 정보 (이미지 박스 오른쪽부터 시작)
        # ---------------------------------------------------------
        info_frame = tk.Frame(card, bg=BG_NORMAL)
        # side='left'로 붙이면, 앞서 만든 220px짜리 박스 바로 뒤에 붙습니다.
        info_frame.pack(side='left', fill='both', expand=True, padx=(40, 10)) 

        # (1) 이름 (위에서 50px 내림)
        lbl_name = tk.Label(info_frame, text=wine.get('name', 'Unknown'), 
                            font=('Helvetica', 20, 'bold'), 
                            bg=BG_NORMAL, fg=FG_PRIMARY, anchor='w')
        lbl_name.pack(fill='x', pady=(70, 2)) 
        
        # (2) 와이너리
        winery = wine.get('winery', 'Unknown Winery')
        lbl_winery = tk.Label(info_frame, text=winery, 
                              font=('Arial', 13, 'bold'), 
                              bg=BG_NORMAL, fg='#dddddd', anchor='w')
        lbl_winery.pack(fill='x', pady=(0, 2))

        # (3) 지역
        raw_region = wine.get('region', [])
        country = wine.get('country', '')
        region_text = ""
        if isinstance(raw_region, list):
            region_text = " / ".join(raw_region) if raw_region else country
        else:
            region_text = f"{raw_region}, {country}" if country else str(raw_region)

        lbl_region = tk.Label(info_frame, text=f"📍 {region_text}", 
                            font=('Arial', 11), bg=BG_NORMAL, fg='#aaaaaa', anchor='w')
        lbl_region.pack(fill='x', pady=(0, 2))

        # (4) 품종
        raw_grapes = wine.get('grapes', [])
        grapes_text = ", ".join(raw_grapes) if isinstance(raw_grapes, list) else str(raw_grapes)
        if not grapes_text: grapes_text = "Unknown Grapes"

        lbl_grapes = tk.Label(info_frame, text=f"🍇 {grapes_text}", 
                              font=('Arial', 11), bg=BG_NORMAL, fg='#999999', anchor='w')
        lbl_grapes.pack(fill='x', pady=(0, 2))

        # (5) 스타일/도수
        style = wine.get('wine_style', '-')
        alcohol = wine.get('alcohol', '-')
        if not style: style = "-"
        if not alcohol: alcohol = "-"
        
        lbl_bottom = tk.Label(info_frame, text=f"{style}   |   💧 {alcohol}", 
                              font=('Arial', 10), bg=BG_NORMAL, fg='#777777', anchor='w')
        lbl_bottom.pack(fill='x')

        # (6) [수정] 실제 카운트 표시
        review_text_color = '#ff5555' if is_disabled else FG_SECONDARY # 부족하면 빨간색/어두운색
        lbl_reviews = tk.Label(info_frame, text=f"💬 Cleaned Reviews: {review_count} {'(Not enough data)' if is_disabled else ''}",
                               font=('Arial', 10, 'italic'), bg=BG_NORMAL, fg=review_text_color, anchor='w')
        lbl_reviews.pack(fill='x', pady=(2, 2))

        # ---------------------------------------------------------
        # 5. 별점
        # ---------------------------------------------------------
        rating_frame = tk.Frame(card, bg=BG_NORMAL)
        rating_frame.pack(side='right', padx=30)
        
        rating = wine.get('rating', 0.0)
        rating_color = "#555555" if is_disabled else "#b89920" # 비활성화면 별점도 회색
        lbl_rating = tk.Label(rating_frame, text=f"★ {rating}", 
                              font=('Arial', 15, 'bold'), bg=BG_NORMAL, fg=rating_color)
        lbl_rating.pack()

        # 이벤트 바인딩 (순서 중요: lbl_reviews 등이 정의된 후)
        if not is_disabled:
            # 배경색 변경 대상들
            bg_targets = [card, img_container, info_frame, lbl_name, lbl_winery, lbl_region, lbl_grapes, lbl_reviews, rating_frame, lbl_rating, lbl_bottom]
            if img_widget: bg_targets.append(img_widget)
            
            # 클릭 이벤트 대상들 (전체)
            all_widgets = bg_targets + [img_widget]

            def on_enter(e):
                # 1. 배경색 밝게
                for w in bg_targets: 
                    try: w.configure(bg=BG_HOVER)
                    except: pass
                # 2. [추가] 하단 텍스트(lbl_bottom) 글자색 밝게 변경!
                lbl_bottom.configure(fg=BOTTOM_TEXT_HOVER) 

            def on_leave(e):
                # 1. 배경색 복구
                for w in bg_targets: 
                    try: w.configure(bg=BG_NORMAL)
                    except: pass
                # 2. [추가] 하단 텍스트 글자색 원래대로 복구
                lbl_bottom.configure(fg=BOTTOM_TEXT_COLOR)
            
            for w in all_widgets:
                if w:
                    w.bind("<Enter>", on_enter)
                    w.bind("<Leave>", on_leave)
                    w.bind("<Button-1>", on_click)

    def _draw_graph(self, wine_data):
        # 공용 함수 호출! (내 분석기, 와인정보, 그리고 내 그래프 프레임을 넘김)
        draw_wine_graph_on_frame(self.analyzer, wine_data, self.graph_frame)

class CategoryTab(ttk.Frame):
    def __init__(self, parent, metadata, analyzer):
        super().__init__(parent)
        self.metadata = metadata
        self.analyzer = analyzer
        self.current_filtered_wines = []
        
        # 저장할 파일명 정의
        self.CATEGORY_DB_FILE = resource_path("winery_category_map.json")
        
        # [핵심 1] 카테고리 DB 파일이 없으면 메타데이터를 분석해서 새로 만듭니다.
        if not os.path.exists(self.CATEGORY_DB_FILE):
            self.generate_category_db_from_metadata()
            
        # [핵심 2] 생성된(혹은 기존의) DB를 로드합니다.
        self.winery_master_db = self.load_category_db()

        # --- UI 초기화 (기존과 동일) ---
        self.paned = tk.PanedWindow(self, orient='horizontal', bg='#1e1e1e', sashwidth=4)
        self.paned.pack(fill='both', expand=True)

        self.left_frame = tk.Frame(self.paned, bg='#2d2d2d', width=300)
        self.paned.add(self.left_frame)
        self._init_tree_view()

        self.right_main_frame = tk.Frame(self.paned, bg='#1e1e1e')
        self.paned.add(self.right_main_frame)
        self.list_view = tk.Frame(self.right_main_frame, bg='#1e1e1e')
        self.detail_view = tk.Frame(self.right_main_frame, bg='#1e1e1e')
        self.list_view.pack(fill='both', expand=True)
        self._init_list_view()
        self._init_detail_view()

        # [핵심 3] 로드된 DB로 트리를 그립니다.
        self.build_category_tree()
        
    # -------------------------------------------------------------------------
    # [1] 메타데이터 분석 및 JSON 생성 (Builder)
    # -------------------------------------------------------------------------
    def generate_category_db_from_metadata(self):
        """
        [최종 로직]
        1. 기존 수동 수정 사항 보존 (JSON에 이미 있는 와이너리는 건너뜀)
        2. 리뷰가 가장 많은 와인을 대표로 선정
        3. 세부 지역명이 'cru'로 끝나면 해당 단계는 카테고리에서 제외 (등급 정보 필터링)
        """
        print("📊 Updating category DB (Cru Filter + Manual Preservation)...")
        
        # 1. 기존 데이터 로드 (수동 수정본 보호용)
        master_db = self.load_category_db()
        existing_wineries = set(master_db.keys())
        
        # 2. 메타데이터 그룹화 (새로 추가할 와이너리만 대상)
        winery_groups = {}
        for wine in self.metadata:
            winery_real = wine.get('winery')
            if not winery_real: continue
            
            winery_key = winery_real.lower().strip()
            
            # 이미 JSON에 등록된 와이너리는 사용자가 수정한 것으로 간주하여 건너뜀
            if winery_key in existing_wineries:
                continue
                
            if winery_key not in winery_groups:
                winery_groups[winery_key] = []
            winery_groups[winery_key].append(wine)

        if not winery_groups:
            print("✨ No new wineries to add. All manual edits are safe.")
            return

        # 3. 새로운 와이너리별 대표 선정 및 경로 최적화
        new_added_count = 0
        for winery_key, wines in winery_groups.items():
            best_wine = None
            max_reviews = 0
            found_high_rating = False

            # 리뷰 수(1순위)와 지역 상세도(2순위)로 대표 와인 선정
            for wine in wines:
                v_info = wine.get('vintage', {})
                count = v_info.get('reviews_count', 0) if isinstance(v_info, dict) else 0
                rating = float(wine.get('rating', 0) or 0.0)
                region_list = wine.get('region', [])
                
                if rating <= 4.0 and rating >=3.8 and not found_high_rating:
                    found_high_rating = True
                    max_reviews = count
                    best_wine = wine
                
                # 2. 이미 4.0 이상인 와인이 있는 상태에서, 더 리뷰가 많은 4.0 이상 와인 발견
                elif rating <= 4.0 and rating >=3.8 and found_high_rating:
                    if count > max_reviews:
                        max_reviews = count
                        best_wine = wine
                
                # 3. 아직 4.0 이상을 못 찾았을 때, 일반 와인들 중 리뷰가 가장 많은 것 유지 (백업)
                elif not found_high_rating:
                    if count > max_reviews:
                        max_reviews = count
                        best_wine = wine

            # 4. [핵심] 경로 생성 및 'Cru' 필터링
            if best_wine:
                country = best_wine.get('country', 'Unknown')
                regions = best_wine.get('region', [])
                
                if not isinstance(regions, list):
                    regions = [regions] if regions else []

                # --- Cru 필터링 로직 추가 ---
                # 마지막 세부 지역명이 'cru'로 끝나면 해당 항목 제거
                if regions:
                    last_region_name = str(regions[-1]).strip().lower()
                    if last_region_name.endswith('cru'):
                        regions = regions[:-1] # 마지막 요소 제외

                # 최종 경로 구성: [Country, Region1, Region2...]
                path = [country]
                for r in regions:
                    if str(r).lower().strip() != country.lower():
                        path.append(str(r).strip())
                
                master_db[winery_key] = path
                new_added_count += 1

        # 5. 결과 저장 (수동 수정본 + 신규 분석본 병합)
        try:
            with open(self.CATEGORY_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(master_db, f, indent=4, ensure_ascii=False)
            print(f"✅ Success: {new_added_count} new wineries added. Cru filtered.")
        except Exception as e:
            print(f"❌ Failed to save category DB: {e}")

    def load_category_db(self):
        """저장된 JSON 파일을 불러옵니다."""
        if not os.path.exists(self.CATEGORY_DB_FILE):
            return {}
        try:
            with open(self.CATEGORY_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading DB: {e}")
            return {}

    # -------------------------------------------------------------------------
    # [2] 트리 구축 (동적 깊이 지원)
    # -------------------------------------------------------------------------
    def build_category_tree(self):
        # 1. 트리 초기화 및 루트 생성
        self.tree.delete(*self.tree.get_children())
        
        self.tree.insert("", "end", "root_style", text="Wine Style", open=False)
        self.tree.insert("", "end", "root_winery", text="Winery", open=False)
        self.tree.insert("", "end", "root_region", text="Region", open=False)

        # ---------------------------------------------------------
        # [STEP 1] Wine Style (기존 로직 유지)
        # ---------------------------------------------------------
        styles = sorted(list(set(str(w.get('wine_style')) for w in self.metadata if w.get('wine_style'))))
        for s in styles:
            self.tree.insert("root_style", "end", text=s, values=("style", s))

        # ---------------------------------------------------------
        # [STEP 2] Winery (원본 코드 로직 그대로 유지)
        # ---------------------------------------------------------
        created_winery_nodes = {}
        my_wineries_meta = [w for w in self.metadata if w.get('winery')]
        my_wineries_meta.sort(key=lambda x: x.get('winery', '').lower())
        processed_wineries = set()

        for wine_obj in my_wineries_meta:
            winery_name_real = wine_obj.get('winery')
            winery_key = winery_name_real.lower().strip()
            if winery_key in processed_wineries: continue

            # [원본 로직] master_db를 참조하고 없으면 Unknown 처리
            path = self.winery_master_db.get(winery_key, ["Unknown"])
            current_parent = "root_winery"
            
            for folder_name in path:
                safe_name = "".join(c for c in folder_name if c.isalnum())
                node_id = f"winery_path_{current_parent}_{safe_name}"
                if not self.tree.exists(node_id):
                    self.tree.insert(current_parent, "end", node_id, text=folder_name, values=("folder", folder_name), open=False)
                current_parent = node_id

            w_id = f"winery_leaf_{winery_key}"
            self.tree.insert(current_parent, "end", w_id, text=winery_name_real, values=("winery", winery_name_real))
            processed_wineries.add(winery_key)

        # ---------------------------------------------------------
        # [STEP 3] Region (지능형 복구 로직 적용)
        # ---------------------------------------------------------
        VALID_COUNTRIES = ["Italy", "France", "Germany", "Spain", "United States", "USA", "Australia", "Chile", "Portugal"]
        region_path_map = {}

        # 1. 정상 와인으로 지역 경로 지도(Map) 생성
        for wine in self.metadata:
            c = wine.get('country', 'Unknown')
            if c in VALID_COUNTRIES:
                regs = wine.get('region', [])
                if not isinstance(regs, list): regs = [regs] if regs else []
                full_p = [c] + [str(r).strip() for r in regs if str(r).strip().lower() != c.lower()]
                for i in range(1, len(full_p)):
                    region_path_map[full_p[i]] = full_p[:i]

        # 2. 모든 와인을 돌며 Region 트리 구축 (모호한 데이터는 지도 참조)
        for wine in self.metadata:
            c = wine.get('country', 'Unknown')
            regs = wine.get('region', [])
            if not isinstance(regs, list): regs = [regs] if regs else []
            
            corrected_path = None
            if c not in VALID_COUNTRIES:
                for key in [c] + regs:
                    if key in region_path_map:
                        corrected_path = region_path_map[key] + [key]
                        break
            
            if not corrected_path:
                if c in VALID_COUNTRIES:
                    corrected_path = [c] + [str(r).strip() for r in regs if str(r).strip().lower() != c.lower()]
                else:
                    corrected_path = ["Unknown", c]

            current_reg_parent = "root_region"
            for r_name in corrected_path:
                safe_r = "".join(c for c in r_name if c.isalnum())
                # ID에 부모 ID를 포함시켜 소지역 이탈 방지
                node_id = f"reg_path_{current_reg_parent}_{safe_r}"
                if not self.tree.exists(node_id):
                    self.tree.insert(current_reg_parent, "end", node_id, text=r_name, values=("region_filter", r_name), open=False)
                current_reg_parent = node_id    # -------------------------------------------------------------------------
    # [3] UI 초기화 메서드들 (기존 유지)
    # -------------------------------------------------------------------------
    def _init_tree_view(self):
        style = ttk.Style()
        style.configure("Treeview", background="#2d2d2d", foreground="white", fieldbackground="#2d2d2d", font=('Arial', 11), rowheight=25)
        style.map('Treeview', background=[('selected', '#555555')])

        self.tree = ttk.Treeview(self.left_frame, show='tree', selectmode='browse')
        self.tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        vsb = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y', pady=10)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def _init_list_view(self):
        header = tk.Frame(self.list_view, bg='#1e1e1e', height=50)
        header.pack(fill='x', padx=20, pady=10)
        self.lbl_category_title = tk.Label(header, text="Select a Category", font=('Helvetica', 18, 'bold'), bg='#1e1e1e', fg='white')
        self.lbl_category_title.pack(side='left')
        
        # DB 재생성 버튼 (숨겨진 기능처럼 작게 추가)
        btn_refresh = tk.Button(header, text="↻ Refresh DB", font=('Arial', 9), bg='#333333', fg='white', relief='flat', 
                                command=self.refresh_database)
        btn_refresh.pack(side='right')

        self.result_area = ScrollableFrame(self.list_view)
        self.result_area.pack(fill='both', expand=True, padx=20, pady=10)
        
        original_scroll_command = self.result_area.scrollbar.set
        def on_scroll_detection(first, last):
            original_scroll_command(first, last)
            if float(last) > 0.9: self.trigger_infinite_scroll()
        self.result_area.canvas.configure(yscrollcommand=on_scroll_detection)

    def _init_detail_view(self):
        nav_frame = tk.Frame(self.detail_view, bg='#1e1e1e', pady=10)
        nav_frame.pack(side='top', fill='x')
        btn_back = ttk.Button(nav_frame, text="⬅ Back to List", command=self.show_list_view)
        btn_back.pack(side='left', padx=20)
        self.lbl_detail_title = tk.Label(nav_frame, text="", font=('Helvetica', 18, 'bold'), bg='#1e1e1e', fg='white')
        self.lbl_detail_title.pack(side='left', padx=20)
        self.graph_frame = tk.Frame(self.detail_view, bg='#1e1e1e')
        self.graph_frame.pack(side='bottom', fill='both', expand=True)

    def refresh_database(self):
        """수동으로 DB를 다시 만들고 트리를 갱신합니다."""
        self.generate_category_db_from_metadata()
        self.winery_master_db = self.load_category_db()
        self.build_category_tree()
        print("Database Refreshed!")

    # -------------------------------------------------------------------------
    # [4] 이벤트 핸들링 (클릭 & 스크롤)
    # -------------------------------------------------------------------------
    def on_tree_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items: return
        item_id = selected_items[0]
        
        item_data = self.tree.item(item_id)
        values = item_data.get('values')
        if not values: return

        filter_type, filter_value = values[0], str(values[1])
        filtered_wines = []

        # [Winery 및 Style] 기존 로직 유지
        if filter_type == 'style':
            filtered_wines = [w for w in self.metadata if str(w.get('wine_style')) == filter_value]
            self.lbl_category_title.config(text=f"Style: {filter_value}")
        elif filter_type == 'winery':
            filtered_wines = [w for w in self.metadata if str(w.get('winery')) == filter_value]
            self.lbl_category_title.config(text=f"Winery: {filter_value}")

        # [Region 전용 필터링] 세부 산지 격리 로직
        elif filter_type == 'region_filter':
            parent_id = self.tree.parent(item_id)
            is_country_node = (parent_id == "root_region")

            for w in self.metadata:
                w_country = w.get('country', '')
                w_regions = w.get('region', [])
                if not isinstance(w_regions, list): w_regions = [w_regions] if w_regions else []
                
                if is_country_node:
                    # 국가 클릭 시 해당 국가 모든 와인
                    if w_country == filter_value:
                        filtered_wines.append(w)
                else:
                    # 세부 지역 클릭 시: 해당 산지가 리스트의 '마지막'인 와인만 (Bourgogne 문제 해결)
                    if w_regions and str(w_regions[-1]).strip() == filter_value:
                        filtered_wines.append(w)
                    # 만약 국가명이 잘못 기재되어 country 필드에 지역명이 들어간 경우도 체크
                    elif not w_regions and w_country == filter_value:
                        filtered_wines.append(w)

            self.lbl_category_title.config(text=f"Region: {filter_value}")

        self.update_wine_list(filtered_wines)
        self.show_list_view()    # -------------------------------------------------------------------------
    # [5] 로딩 및 무한 스크롤 (이전과 동일)
    # -------------------------------------------------------------------------
    def update_wine_list(self, wines):
        if hasattr(self, 'loading_task') and self.loading_task:
            self.after_cancel(self.loading_task)
            self.loading_task = None
        for widget in self.result_area.scrollable_frame.winfo_children(): widget.destroy()
        if not wines:
            tk.Label(self.result_area.scrollable_frame, text="No wines found.", bg='#1e1e1e', fg='gray', font=('Arial', 14)).pack(pady=50)
            return
        self.result_area.canvas.yview_moveto(0)
        self.current_wine_queue = wines
        self.loaded_count = 0
        self.is_loading = False
        self.load_next_chunk()

    def trigger_infinite_scroll(self):
        """스크롤이 바닥에 닿았을 때 호출됨"""
        # [수정] 안전장치 추가: 아직 데이터 큐가 생성되지 않았으면 무시
        if not hasattr(self, 'current_wine_queue') or not self.current_wine_queue:
            return

        # 1. 이미 로딩 중이면 무시 (중복 실행 방지)
        if getattr(self, 'is_loading', False):
            return
        
        # 2. 더 불러올 데이터가 없으면 무시
        if self.loaded_count >= len(self.current_wine_queue):
            return

        # 3. 로딩 시작
        self.load_next_chunk()

    def load_next_chunk(self):
        self.is_loading = True
        start_index = self.loaded_count
        PAGE_SIZE = 50
        batch_data = self.current_wine_queue[start_index : min(start_index + PAGE_SIZE, len(self.current_wine_queue))]
        if not batch_data:
            self.is_loading = False
            return
        self.render_batch_internal(batch_data, 0)

    def render_batch_internal(self, batch_data, local_index):
        MINI_BATCH = 10
        chunk = batch_data[local_index : min(local_index + MINI_BATCH, len(batch_data))]
        for wine in chunk: self.create_wine_card(wine)
        if local_index + MINI_BATCH < len(batch_data):
            self.loading_task = self.after(10, lambda: self.render_batch_internal(batch_data, local_index + MINI_BATCH))
        else:
            self.loaded_count += len(batch_data)
            self.is_loading = False
            self.loading_task = None

    # -------------------------------------------------------------------------
    # [6] 카드 생성 및 상세 (이전과 동일)
    # -------------------------------------------------------------------------
    def create_wine_card(self, wine):
        wine_id = wine.get('id')
        if 'cached_review_count' in wine: review_count = wine['cached_review_count']
        else:
            file_path = os.path.join("cleaned", f"wine_{wine_id}_clean.jsonl")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: review_count = sum(1 for _ in f)
                except: review_count = 0
            else: review_count = 0
            wine['cached_review_count'] = review_count
            
        is_disabled = review_count <= 3
        BG_NORMAL = '#222222' if is_disabled else '#333333'
        BG_HOVER = '#222222' if is_disabled else '#3e3e3e'
        FG_PRIMARY = '#555555' if is_disabled else 'white'
        FG_SECONDARY = '#444444' if is_disabled else '#aaaaaa'
        
        card = tk.Frame(self.result_area.scrollable_frame, bg=BG_NORMAL, bd=0, height=300, cursor='arrow' if is_disabled else 'hand2')
        card.pack(fill='x', pady=8, padx=5, ipady=0); card.pack_propagate(False)

        def on_click(e):
            if not is_disabled: self.show_detail_view(wine)

        img_container = tk.Frame(card, bg=BG_NORMAL, width=300, height=300); img_container.pack(side='left', fill='y'); img_container.pack_propagate(False)
        
        has_image = wine.get('image', 0)
        image_path = os.path.join("image", f"wine_{wine_id}_image.png")
        img_widget = None
        if has_image == 1 and os.path.exists(image_path):
            try:
                pil_img = Image.open(image_path).convert("RGBA")
                orig_w, orig_h = pil_img.size
                if orig_h > 50: cropped_img = pil_img.crop((0, int(orig_h*0.4), orig_w, int(orig_h*0.95)))
                else: cropped_img = pil_img
                ar = cropped_img.size[0]/cropped_img.size[1]
                final_img = cropped_img.resize((int(300*ar), 300), Image.Resampling.LANCZOS)
                bg = Image.new('RGBA', final_img.size, (int(self.winfo_rgb(BG_NORMAL)[0]/256), int(self.winfo_rgb(BG_NORMAL)[1]/256), int(self.winfo_rgb(BG_NORMAL)[2]/256), 255))
                final_img = Image.alpha_composite(bg, final_img).convert("RGB").filter(ImageFilter.SHARPEN)
                if is_disabled: final_img = ImageEnhance.Brightness(final_img).enhance(0.3)
                tk_img = ImageTk.PhotoImage(final_img)
                img_widget = tk.Label(img_container, image=tk_img, bg=BG_NORMAL, bd=0); img_widget.image = tk_img
            except: pass
        
        if not img_widget:
            tk_ph = ImageTk.PhotoImage(Image.new('RGB', (80, 300), '#222222' if is_disabled else '#444444'))
            img_widget = tk.Label(img_container, image=tk_ph, text="No\nImg", fg=FG_PRIMARY, bg=BG_NORMAL, compound='center'); img_widget.image=tk_ph
        img_widget.place(relx=0.5, rely=0.5, anchor='center')

        info_frame = tk.Frame(card, bg=BG_NORMAL); info_frame.pack(side='left', fill='both', expand=True, padx=(10, 0))
        tk.Label(info_frame, text=wine.get('name', 'Unknown'), font=('Helvetica', 20, 'bold'), bg=BG_NORMAL, fg=FG_PRIMARY, anchor='w').pack(fill='x', pady=(70, 2))
        tk.Label(info_frame, text=wine.get('winery', 'Unknown Winery'), font=('Arial', 13, 'bold'), bg=BG_NORMAL, fg='#dddddd', anchor='w').pack(fill='x', pady=(0, 2))
        
        raw_r = wine.get('region', []); r_txt = " / ".join(raw_r) if isinstance(raw_r, list) else str(raw_r)
        tk.Label(info_frame, text=f"📍 {r_txt}", font=('Arial', 11), bg=BG_NORMAL, fg='#aaaaaa', anchor='w').pack(fill='x', pady=(0, 2))
        
        raw_g = wine.get('grapes', []); g_txt = ", ".join(raw_g) if isinstance(raw_g, list) else str(raw_g)
        tk.Label(info_frame, text=f"🍇 {g_txt}", font=('Arial', 11), bg=BG_NORMAL, fg='#999999', anchor='w').pack(fill='x', pady=(0, 2))
        
        lbl_btm = tk.Label(info_frame, text=f"{wine.get('wine_style', '-')} | 💧 {wine.get('alcohol', '-')}", font=('Arial', 10), bg=BG_NORMAL, fg='#777777', anchor='w'); lbl_btm.pack(fill='x')
        tk.Label(info_frame, text=f"💬 Reviews: {review_count}", font=('Arial', 10, 'italic'), bg=BG_NORMAL, fg='#ff5555' if is_disabled else FG_SECONDARY, anchor='w').pack(fill='x', pady=(2, 2))

        rf = tk.Frame(card, bg=BG_NORMAL); rf.pack(side='right', padx=30)
        tk.Label(rf, text=f"★ {wine.get('rating', 0.0)}", font=('Arial', 15, 'bold'), bg=BG_NORMAL, fg="#555555" if is_disabled else "#b89920").pack()

        if not is_disabled:
            targets = [card, img_container, info_frame, lbl_btm, rf, img_widget]
            for c in info_frame.winfo_children(): targets.append(c)
            for c in rf.winfo_children(): targets.append(c)
            def on_e(e): 
                for w in targets: 
                    try: w.configure(bg=BG_HOVER)
                    except: pass
                lbl_btm.configure(fg='#bbbbbb')
            def on_l(e): 
                for w in targets: 
                    try: w.configure(bg=BG_NORMAL)
                    except: pass
                lbl_btm.configure(fg='#777777')
            for w in targets: w.bind("<Enter>", on_e); w.bind("<Leave>", on_l); w.bind("<Button-1>", on_click)

    def show_list_view(self):
        self.detail_view.pack_forget(); self.list_view.pack(fill='both', expand=True)

    def show_detail_view(self, wine_data):
        self.list_view.pack_forget(); self.detail_view.pack(fill='both', expand=True)
        self.lbl_detail_title.config(text=wine_data.get('name', 'Unknown'))
        self._draw_graph(wine_data)

    def _draw_graph(self, wine_data):
        # 공용 함수 호출! (내 분석기, 와인정보, 그리고 내 그래프 프레임을 넘김)
        draw_wine_graph_on_frame(self.analyzer, wine_data, self.graph_frame)

class AnalyticsTab(ttk.Frame):
    def __init__(self, parent, metadata, analyzer):
        super().__init__(parent)
        self.metadata = metadata
        self.analyzer = analyzer
        self.current_recommendations = [] 
        self.target_wine_id = None        
        
        # ---------------------------------------------------------
        # 메인 레이아웃: 좌우 2분할
        # ---------------------------------------------------------
        self.paned = tk.PanedWindow(self, orient='horizontal', bg='#1e1e1e', sashwidth=4)
        self.paned.pack(fill='both', expand=True)

        # [왼쪽] 검색창 (너비 고정 400px)
        self.left_frame = tk.Frame(self.paned, bg='#2d2d2d', width=400)
        self.paned.add(self.left_frame)
        self.left_frame.pack_propagate(True)

        # [오른쪽] 상단 고정 타겟 + 하단 스크롤 리스트
        self.right_frame = tk.Frame(self.paned, bg='#1e1e1e')
        self.paned.add(self.right_frame)

        # =========================================================
        # [왼쪽 UI] 검색
        # =========================================================
        search_box = tk.Frame(self.left_frame, bg='#333333', height=50)
        search_box.pack(fill='x', padx=15, pady=(20, 10))
        
        self.search_var = tk.StringVar()
        entry = tk.Entry(search_box, textvariable=self.search_var, font=('Arial', 14), 
                         bg='#333333', fg='white', relief='flat', insertbackground='white')
        entry.pack(fill='both', expand=True, padx=10, pady=10)
        entry.bind("<Return>", self.perform_search)
        
        self.lbl_guide = tk.Label(self.left_frame, text="🔍 Search & Select a wine", 
                                  font=('Arial', 11), fg='#666666', bg='#2d2d2d')
        self.lbl_guide.pack(pady=10)

        self.search_results_area = ScrollableFrame(self.left_frame)
        self.search_results_area.pack(fill='both', expand=True, padx=10, pady=10)

        # [해결책] 생성된 스크롤바의 두께를 명시적으로 다시 지정합니다.
        # 다른 곳에서는 기본값을 쓰더라도, 여기서는 20px로 강제합니다.
        try:
            self.search_results_area.scrollbar.configure(width=20)
        except:
            pass

        # =========================================================
        # [오른쪽 UI] 1. 상단 고정 타겟 영역
        # =========================================================
        self.fixed_target_frame = tk.Frame(self.right_frame, bg='#1e1e1e', height=200)
        
        # [수정 1] pady=(20, 0) -> 아래쪽 여백을 0으로 제거 (텍스트와 붙임)
        self.fixed_target_frame.pack(side='top', fill='x', padx=(20, 37), pady=(20, 0))
        
        self.fixed_target_frame.pack_propagate(False)

        self.lbl_target_placeholder = tk.Label(self.fixed_target_frame, 
                                               text="Selected Target Wine will appear here", 
                                               font=('Arial', 14), fg='#555555', bg='#1e1e1e')
        self.lbl_target_placeholder.place(relx=0.5, rely=0.5, anchor='center')

        # =========================================================
        # [오른쪽 UI] 2. 중간 텍스트 (Comparing with...)
        # =========================================================
        self.header_lbl = tk.Label(self.right_frame, text="Similar Wines", 
                                   font=('Helvetica', 16, 'bold'), bg='#1e1e1e', fg='#aaaaaa')
        
        # [수정 2] pady=(2, 2) -> 위아래 여백을 최소화하여 공간을 줄임
        self.header_lbl.pack(pady=(2, 2))

        # =========================================================
        # [오른쪽 UI] 3. 하단 추천 리스트
        # =========================================================
        self.rec_area = ScrollableFrame(self.right_frame)
        
        # [수정 3] pady=(0, 10) -> 위쪽 여백을 0으로 제거 (텍스트와 붙임)
        self.rec_area.pack(fill='both', expand=True, padx=20, pady=(0, 10))

        # --- [추가] 로딩 오버레이 레이어 ---
        self.loading_overlay = tk.Frame(self.right_frame, bg='#1e1e1e')
        # 초기에는 숨겨둠
        
        self.loading_label = tk.Label(self.loading_overlay, text="Analyzing .", 
                                      font=('Helvetica', 22, 'bold'), 
                                      fg='#ffffff', bg='#1e1e1e')
        self.loading_label.place(relx=0.5, rely=0.5, anchor='center')
        self.dot_count = 1  

        # 무한 스크롤 연결
        original_scroll = self.rec_area.scrollbar.set
        def on_scroll(first, last):
            original_scroll(first, last)
            if float(last) > 0.9: self.trigger_infinite_scroll()
        self.rec_area.canvas.configure(yscrollcommand=on_scroll)

    def animate_loading(self):
        """별도 스레드에서 실행되어 메인 렉에 영향을 받지 않는 로딩 애니메이션"""
        def run():
            self.dot_count = 1
            while getattr(self, 'is_loading_ui', False):
                dots = "." * self.dot_count
                # UI 업데이트는 thread-safe하게 config로 전달
                try:
                    self.loading_label.config(text=f"Analyzing{dots}")
                except:
                    break
                self.dot_count = (self.dot_count % 3) + 1
                time.sleep(0.5) # 애니메이션 속도
        
        # 데몬 스레드로 실행 (프로그램 종료 시 같이 종료)
        threading.Thread(target=run, daemon=True).start()

    def show_loading(self, show=True):
        if show:
            # 이미 로딩 중이면 새 애니메이션 스레드를 만들지 않음
            if getattr(self, 'is_loading_ui', False): return 
            
            self.is_loading_ui = True
            self.loading_overlay.place(in_=self.rec_area, relx=0, rely=0, relwidth=1, relheight=1)
            self.loading_overlay.lift()
            self.animate_loading()
        else:
            self.is_loading_ui = False
            self.loading_overlay.place_forget()

    def create_card_widget(self, parent, wine, score=None, is_target=False):
        """
        와인 카드를 생성해서 반환합니다.
        is_target=True 이면 배경색과 뱃지가 다르게 적용됩니다.
        """
        CARD_HEIGHT = 220
        # 타겟 와인은 조금 더 밝은 배경으로 구분
        BG_COLOR = '#1e1e1e' if is_target else "#2c2c2c"
        
        card = tk.Frame(parent, bg=BG_COLOR, bd=0, height=CARD_HEIGHT, cursor='hand2')
        card.pack_propagate(False) 

        # 클릭 시 타겟 변경
        def on_click(e): 
            if not is_target: self.set_target_wine(wine)

        # 1. 뱃지 (왼쪽)
        if is_target:
            badge_color = "#3F51B5" # 타겟은 파란색 계열
            main_text = "TARGET"
            sub_text = "Standard"
        else:
            score_percent = int((score or 0) * 100)
            badge_color = "#4CAF50" if score_percent >= 70 else ("#FFC107" if score_percent >= 40 else "#FF5722")
            main_text = f"{score_percent}%"
            sub_text = "Match"

        score_frame = tk.Frame(card, bg=badge_color, width=70)
        score_frame.pack(side='left', fill='y')
        score_frame.pack_propagate(False)
        
        tk.Label(score_frame, text=main_text, font=('Arial', 11 if is_target else 14, 'bold'), 
                 bg=badge_color, fg='white').pack(expand=True)
        tk.Label(score_frame, text=sub_text, font=('Arial', 8), 
                 bg=badge_color, fg='white').pack(pady=(0, 20))

        # 2. 이미지 (왼쪽 여백 50px)
        img_container = tk.Frame(card, bg=BG_COLOR, width=110)
        img_container.pack(side='left', fill='y', padx=(50, 20))
        img_container.pack_propagate(False)
        
        wine_id = wine.get('id')
        img_path = os.path.join("image", f"wine_{wine_id}_image.png")
        img_widget = None

        if os.path.exists(img_path):
            try:
                from PIL import ImageEnhance, ImageFilter
                pil_img = Image.open(img_path).convert("RGBA")
                orig_w, orig_h = pil_img.size
                if orig_h > 50:
                    top_cut = int(orig_h * 0.40); bottom_cut = int(orig_h * 0.95)
                    cropped_img = pil_img.crop((0, top_cut, orig_w, bottom_cut))
                else: cropped_img = pil_img
                
                crop_w, crop_h = cropped_img.size
                aspect = crop_w / crop_h
                new_h = CARD_HEIGHT; new_w = int(new_h * aspect)
                resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                r, g, b = self.winfo_rgb(BG_COLOR)
                bg_tup = (r//256, g//256, b//256, 255)
                bg = Image.new('RGBA', resized.size, bg_tup) 
                final_img = Image.alpha_composite(bg, resized).convert("RGB").filter(ImageFilter.SHARPEN)
                
                tk_img = ImageTk.PhotoImage(final_img)
                img_widget = tk.Label(img_container, image=tk_img, bg=BG_COLOR, bd=0)
                img_widget.image = tk_img
            except: pass
        
        if not img_widget:
             tk_ph = ImageTk.PhotoImage(Image.new('RGB', (80, CARD_HEIGHT), '#444444'))
             img_widget = tk.Label(img_container, image=tk_ph, text="No\nImg", fg='#888888', bg=BG_COLOR)
             img_widget.image = tk_ph
        img_widget.place(relx=0.5, rely=0.5, anchor='center')

        # 3. 미니 그래프 (우측 끝, 타겟 카드에도 표시됨!)
        graph_frame = tk.Frame(card, bg=BG_COLOR, width=350) 
        graph_frame.pack(side='right', fill='y', padx=0)
        graph_frame.pack_propagate(False)

        data_path = os.path.join("data", f"wine_{wine.get('id')}_data.json")
        
        # [수정됨] Analyzer 클래스의 메서드 호출로 간소화
        if os.path.exists(data_path):
            try:
                # 여기서 StreamAnalyzer의 로직을 가져와서 그립니다!
                fig = self.analyzer.create_mini_graph(data_path)
                
                if fig:
                    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
                    cw = canvas.get_tk_widget()
                    # 배경색 통합 및 테두리 제거
                    cw.configure(bg=BG_COLOR, highlightthickness=0, bd=0)
                    cw.pack(fill='both', expand=True, pady=0)
                    
                    canvas.draw()
                    
                    # 클릭 이벤트 바인딩 (그래프 눌러도 타겟 변경되게)
                    # 참고: is_target이 아닐 때만 타겟 변경 함수 연결
                    if not is_target:
                        cw.bind("<Button-1>", lambda e: self.set_target_wine(wine))
            except Exception as e:
                # print(f"Graph Error: {e}") 
                pass

        # 4. 정보 영역 (중간)
        info_frame = tk.Frame(card, bg=BG_COLOR)
        info_frame.pack(side='left', fill='both', expand=True, pady=(40,10))

        def add_lbl(text, font, fg, pady=0):
             tk.Label(info_frame, text=text, font=font, bg=BG_COLOR, fg=fg, 
                      anchor='w').pack(fill='x', pady=pady)

        row1 = tk.Frame(info_frame, bg=BG_COLOR)
        row1.pack(fill='x', pady=(0, 2), anchor='w')
        
        tk.Label(row1, text=wine.get('name', 'Unknown'), font=('Arial', 14, 'bold'), 
                 bg=BG_COLOR, fg='white', anchor='w').pack(side='left')
        
        rating = wine.get('rating', 0.0)
        tk.Label(row1, text=f"   ★ {rating}", font=('Arial', 12, 'bold'), 
                 bg=BG_COLOR, fg="#b89920").pack(side='left')

        add_lbl(f"{wine.get('winery')}  |  {wine.get('country')}", ('Arial', 11), '#aaaaaa')
        
        raw_r = wine.get('region', []); r_txt = " / ".join(raw_r) if isinstance(raw_r, list) else str(raw_r)
        add_lbl(f"📍 {r_txt}", ('Arial', 10), '#888888', pady=(5,0))

        raw_g = wine.get('grapes', []); g_txt = ", ".join(raw_g) if isinstance(raw_g, list) else str(raw_g)
        add_lbl(f"🍇 {g_txt}", ('Arial', 10), '#777777')
        
        style = wine.get('wine_style', '-') or '-'; alcohol = wine.get('alcohol', '-') or '-'
        add_lbl(f"{style} | 💧 {alcohol}", ('Arial', 9), '#777777', pady=(2,0))

        # 리뷰 수
        if 'cached_review_count' in wine:
            review_count = wine['cached_review_count']
        else:
            f_path = os.path.join("cleaned", f"wine_{wine_id}_clean.jsonl")
            review_count = 0
            if os.path.exists(f_path):
                try: 
                    with open(f_path, 'r', encoding='utf-8') as f: review_count = sum(1 for _ in f)
                except: pass
            wine['cached_review_count'] = review_count
        
        rev_color = '#ff5555' if review_count <= 3 else '#666666'
        add_lbl(f"💬 Reviews: {review_count}", ('Arial', 9, 'italic'), rev_color, pady=(3,0))

        # 이벤트 바인딩
        widgets = [card, score_frame, img_container, info_frame, row1, graph_frame] + \
                  list(info_frame.winfo_children()) + list(score_frame.winfo_children()) + list(row1.winfo_children())
        if img_widget: widgets.append(img_widget)
        
        for w in widgets:
            try: w.bind("<Button-1>", on_click)
            except: pass
            
        return card

    def _normalize_text(self, text):
        """
        독일어 움라우트(ö, ü) 및 프랑스어 악센트를 
        영어 알파벳(o, u, a)으로 안전하게 변환하는 정규화 로직
        """
        if text is None: return ""
        text = str(text)
        
        # 1. 유니코드 분해 (NFD: 'ö'를 'o'와 '¨'로 나눔)
        nfd_text = unicodedata.normalize('NFD', text)
        
        # 2. 'Mn'(Mark, Nonspacing) 카테고리(악센트 기호)만 필터링하고 다시 합침
        # 이렇게 하면 'o'는 남고 위쪽의 점 두개(¨)만 사라집니다.
        clean_text = "".join([c for c in nfd_text if unicodedata.category(c) != 'Mn'])
        
        # 3. 소문자 변환 및 공백 제거
        return clean_text.lower().strip()

    def perform_search(self, event=None):
        """[들여쓰기 수정 버전]"""
        # 함수 정의 바로 아랫줄은 반드시 한 단계 들여쓰기가 되어야 합니다.
        raw_query = self.search_var.get().strip()
        query = self._normalize_text(raw_query)
        
        # 이전 결과 삭제
        for widget in self.search_results_area.scrollable_frame.winfo_children():
            widget.destroy()

        if not query:
            return

        found_items = []
        for w in self.metadata:
            name = self._normalize_text(w.get('name', ''))
            winery = self._normalize_text(w.get('winery', ''))
            
            # 검색어 매칭
            if (query in name) or (query in winery):
                found_items.append(w)

        if not found_items:
            tk.Label(self.search_results_area.scrollable_frame, text="No results found.", 
                     bg='#2d2d2d', fg='gray').pack(pady=20)
        else:
            for wine_data in found_items:
                self.create_mini_search_card(wine_data)

        self.search_results_area.canvas.yview_moveto(0)

    def create_mini_search_card(self, wine):
        """검색 결과용 미니 카드 (리뷰 40개 이하 비활성화 로직 추가)"""
        # 리뷰 수 확인
        wine_id = wine.get('id')
        review_count = wine.get('cached_review_count', 0)
        
        # 리뷰 데이터가 캐싱되지 않았다면 여기서 확인
        if review_count == 0:
            f_path = os.path.join("cleaned", f"wine_{wine_id}_clean.jsonl")
            if os.path.exists(f_path):
                try:
                    with open(f_path, 'r', encoding='utf-8') as f:
                        review_count = sum(1 for _ in f)
                except: pass
            wine['cached_review_count'] = review_count

        # [핵심] 비활성화 조건 (40개 이하)
        is_disabled = review_count <= 30
        
        # 스타일에 반영
        bg_color = '#222222' if is_disabled else '#333333'
        fg_name = '#555555' if is_disabled else 'white'
        fg_winery = '#444444' if is_disabled else '#888888'
        cursor_style = 'arrow' if is_disabled else 'hand2'

        card = tk.Frame(self.search_results_area.scrollable_frame, bg=bg_color, cursor=cursor_style)
        card.pack(fill='x', pady=2, padx=(5, 25)) 
        
        name = wine.get('name', 'Unknown')
        winery = wine.get('winery', 'Unknown')
        
        lbl_name = tk.Label(card, text=name, font=('Arial', 10, 'bold'), 
                            bg=bg_color, fg=fg_name, anchor='w', 
                            wraplength=280, justify='left')
        lbl_name.pack(fill='x', padx=10, pady=(8, 2))
        
        # 리뷰 수도 함께 표시하여 사용자에게 이유를 알림
        winery_text = f"{winery} (Reviews: {review_count})"
        lbl_winery = tk.Label(card, text=winery_text, font=('Arial', 9), 
                              bg=bg_color, fg=fg_winery, anchor='w')
        lbl_winery.pack(fill='x', padx=10, pady=(0, 8))

        # [중요] 비활성화 상태가 아닐 때만 이벤트 바인딩
        if not is_disabled:
            def select(e): self.set_target_wine(wine)
            card.bind("<Button-1>", select)
            lbl_name.bind("<Button-1>", select)
            lbl_winery.bind("<Button-1>", select)
        
        return card

    def set_target_wine(self, wine):
        """
        실행 순서: 
        1. 로딩창 표시 및 기존 리스트 초기화 (즉시)
        2. 선택 와인 카드 렌더링 (잠시 후)
        3. 비교 분석 실행 (최종)
        """
        # --- 1단계: 로딩창 표시 및 초기화 ---
        self.show_loading(True)
        
        # 이전 추천 리스트 즉시 삭제
        for widget in self.rec_area.scrollable_frame.winfo_children():
            widget.destroy()
        
        # UI 강제 갱신 (Analyzing... 문구를 먼저 띄움)
        self.update_idletasks()

        # --- 2단계: 0.1초 뒤 선택 와인 카드 렌더링 ---
        # 렉을 줄이기 위해 카드를 그리는 동작을 아주 짧은 시간 뒤로 미룹니다.
        self.after(100, lambda: self._update_target_ui(wine))

    def _update_target_ui(self, wine):
        """상단 타겟 UI를 업데이트하고 분석을 호출하는 내부 함수"""
        self.target_wine_id = wine.get('id')
        
        # 기존 Placeholder 및 카드 제거
        for w in self.fixed_target_frame.winfo_children():
            w.destroy()

        # 새로운 타겟 카드 생성
        target_card = self.create_card_widget(self.fixed_target_frame, wine, is_target=True)
        target_card.pack(fill='both', expand=True)
        
        self.header_lbl.config(text=f"Comparing with: {wine.get('name')}")
        
        # UI 업데이트 후 분석 실행 (3단계)
        self.update_idletasks()
        # 분석 로직 실행 (이 안에서 결과가 나오면 show_loading(False) 호출)
        self.run_similarity_analysis(wine)

    def get_flavor_vector(self, wine_id):
        path = os.path.join("data", f"wine_{wine_id}_data.json")
        if not os.path.exists(path): return None
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            vector = {}
            for flavor, info in data.items():
                if 'w' in info and 'x' in info:
                    # [개선] 향의 강도를 시간대별 가중치로 분산 저장
                    # 예: x=0.2(초반), w=10 이면 'flavor_early'에 점수 부여
                    for x_val, w_val in zip(info['x'], info['w']):
                        if x_val < 0.33: tag = "_early"
                        elif x_val < 0.66: tag = "_mid"
                        else: tag = "_late"
                        
                        key = f"{flavor}{tag}"
                        vector[key] = vector.get(key, 0) + w_val
            return vector
        except: return None

    def calculate_advanced_similarity(self, target_vec, cand_vec):
        """
        코사인 유사도에 '불순물 패널티'를 적용한 점수를 반환합니다.
        Target에 없는 맛을 Candidate가 가지고 있을수록 점수가 깎입니다.
        """
        # 1. 교집합 확인
        common_keys = set(target_vec.keys()) & set(cand_vec.keys())
        if not common_keys: return 0.0
        
        # 2. 기본 코사인 유사도 계산 (벡터 각도)
        dot = sum(target_vec[k] * cand_vec[k] for k in common_keys)
        norm1 = math.sqrt(sum(v**2 for v in target_vec.values()))
        norm2 = math.sqrt(sum(v**2 for v in cand_vec.values()))
        
        if norm1 == 0 or norm2 == 0: return 0.0
        cosine_sim = dot / (norm1 * norm2)
        
        # 3. [핵심] 불순물 패널티 (Alien Flavor Penalty)
        # Candidate가 가진 맛 중에서 Target에는 없는 맛들의 가중치 합을 구함
        cand_total_weight = sum(cand_vec.values())
        alien_weight = sum(cand_vec[k] for k in cand_vec if k not in target_vec)
        
        # 불순물 비율 (0.0 ~ 1.0)
        # 예: 전체 맛이 100인데, 타겟에 없는 맛이 30이면 ratio는 0.3
        alien_ratio = alien_weight / cand_total_weight if cand_total_weight > 0 else 0
        
        # 최종 점수 = 코사인 점수 * (1 - 불순물 비율)
        # 즉, 타겟에 없는 맛이 많을수록 점수가 깎임 (Purity 개념)
        final_score = cosine_sim * (1.0 - alien_ratio)
        
        return final_score

    def check_structure_match(self, target_wine, cand_wine):
        """
        [필터링 1단계]
        Target 와인이 가진 맛 데이터 필드(None이 아닌 것)의 구성이
        Candidate 와인과 '정확히 일치'하는지 검사합니다.
        예: Target이 [Body, Sweet]만 있으면, Candidate도 [Body, Sweet]만 있어야 함.
        """
        keys = ['body_score', 'tannin_score', 'sweetness_score', 'acidity_score']
        
        # 값이 존재하는(None이 아닌) 키들의 집합을 만듦
        target_keys = {k for k in keys if target_wine.get(k) is not None}
        cand_keys = {k for k in keys if cand_wine.get(k) is not None}
        
        # 집합이 정확히 같아야 통과 (Target: 4개, Cand: 3개 -> 탈락)
        return target_keys == cand_keys

    def calculate_structure_similarity(self, target_wine, cand_wine):
        """
        [점수 계산]
        맛 구조 수치(%)의 차이를 계산하여 유사도(0.0 ~ 1.0)를 반환합니다.
        차이가 작을수록 점수가 높습니다.
        """
        keys = ['body_score', 'tannin_score', 'sweetness_score', 'acidity_score']
        valid_keys = [k for k in keys if target_wine.get(k) is not None]
        
        if not valid_keys: return 1.0 # 비교할 데이터가 둘 다 없으면 구조는 같다고 봄
        
        total_diff = 0
        for k in valid_keys:
            v1 = float(target_wine.get(k, 0))
            v2 = float(cand_wine.get(k, 0))
            # 차이의 절댓값 (0 ~ 100)
            diff = abs(v1 - v2)
            total_diff += diff
            
        # 평균 차이 계산
        avg_diff = total_diff / len(valid_keys)
        
        # 유사도 변환: 차이가 0이면 1.0(100%), 차이가 100이면 0.0(0%)
        # 민감도 조절: 차이가 20% 이상이면 꽤 다른 것이므로 감점을 크게 줄 수도 있음.
        # 여기선 선형적으로 계산: 1 - (평균차이 / 100)
        sim = 1.0 - (avg_diff / 100.0)
        return max(0.0, sim)

    def run_similarity_analysis(self, target_wine):
        target_vec = self.get_flavor_vector(target_wine['id'])
        
        # 데이터가 없는 경우 로딩창을 닫고 종료
        if not target_vec:
            print(f"⚠️ 와인 ID {target_wine['id']}의 맛 프로필 데이터가 없습니다.")
            for w in self.rec_area.scrollable_frame.winfo_children(): w.destroy()
            tk.Label(self.rec_area.scrollable_frame, text="Insufficient flavor data for analysis.", 
                     bg='#1e1e1e', fg='gray', font=('Arial', 14)).pack(pady=50)
            self.show_loading(False) # [추가] 로딩창 닫기
            return

        scores = []
        
        for wine in self.metadata:
            # 1. 자기 자신 제외
            if wine['id'] == target_wine['id']: continue
            
            # 2. [NEW] 맛 구조(Body, Tannin 등) 구성 일치 여부 확인 (Strict Filter)
            if not self.check_structure_match(target_wine, wine):
                continue # 구성이 다르면 아예 비교 대상에서 제외
            
            # 3. 향(Flavor) 벡터 가져오기
            cand_vec = self.get_flavor_vector(wine['id'])
            if not cand_vec: continue
            
            # 4. 향(Flavor) 유사도 계산 (90% 컷오프)
            flavor_sim = self.calculate_advanced_similarity(target_vec, cand_vec)
            if flavor_sim < 0.90: continue # 90% 미만 탈락
            
            # 5. [NEW] 맛 구조(Structure) 유사도 계산
            struct_sim = self.calculate_structure_similarity(target_wine, wine)
            
            # 6. 최종 점수 합산 (가중치 적용)
            # Flavor(향) 70% + Structure(구조) 30% 비중으로 합산
            # 향이 더 중요하지만, 바디감이 너무 다르면 안 되므로 구조 점수도 반영
            final_score = (flavor_sim * 0.7) + (struct_sim * 0.3)
            
            scores.append((wine, final_score))
        
        # 점수 높은 순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        self.current_recommendations = scores
        self.update_recommendation_list()

    def update_recommendation_list(self):
        if hasattr(self, 'loading_task') and self.loading_task:
            self.after_cancel(self.loading_task)
            self.loading_task = None
            
        # 기존 리스트 삭제
        for widget in self.rec_area.scrollable_frame.winfo_children(): 
            widget.destroy()

        if not self.current_recommendations:
            self.show_loading(False) # 결과 없으면 로딩 끔
            tk.Label(self.rec_area.scrollable_frame, text="No similar wines found.", 
                     bg='#1e1e1e', fg='gray', font=('Arial', 14)).pack(pady=50)
            return

        # --- 로딩 시작 ---
        self.show_loading(True)
        self.rec_area.canvas.yview_moveto(0)
        self.loaded_count = 0
        self.is_loading = False
        
        # 첫 번째 배치를 로딩 레이어 뒤에서 생성 시작
        self.load_next_chunk()

    def trigger_infinite_scroll(self):
        if not hasattr(self, 'current_recommendations') or not self.current_recommendations: return
        if getattr(self, 'is_loading', False): return
        if self.loaded_count >= len(self.current_recommendations): return
        self.load_next_chunk()

    def load_next_chunk(self):
        self.is_loading = True
        start_index = self.loaded_count
        PAGE_SIZE = 50
        batch_data = self.current_recommendations[start_index : min(start_index + PAGE_SIZE, len(self.current_recommendations))]
        if not batch_data:
            self.is_loading = False
            return
        self.render_batch_internal(batch_data, 0)

    def render_batch_internal(self, batch_data, local_index):
        # [수정] 한 번에 그리는 양을 2~3개로 대폭 축소
        MINI_BATCH = 2 
        chunk = batch_data[local_index : min(local_index + MINI_BATCH, len(batch_data))]
        
        for wine, score in chunk:
            card = self.create_card_widget(self.rec_area.scrollable_frame, wine, score, is_target=False)
            card.pack(fill='x', pady=5, padx=5)

        if local_index + MINI_BATCH < len(batch_data):
            # [수정] 다음 카드 그리기 전 대기 시간을 50ms 정도로 늘려 UI가 반응할 시간을 줌
            self.loading_task = self.after(50, lambda: self.render_batch_internal(batch_data, local_index + MINI_BATCH))
        else:
            self.loaded_count += len(batch_data)
            self.is_loading = False
            self.loading_task = None
            self.show_loading(False)

class WineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vivino Flavor Studio v7.0")
        self.root.geometry("1920x1080")
        self.root.configure(bg='#1e1e1e')

        self.wine_metadata = self.load_metadata(resource_path("wine_metadata.jsonl"))
        self.analyzer = WineStreamAnalyzer()
        self._init_ui()

    def load_metadata(self, filepath):
        # ... (이전 대화의 "강력한 로더" 코드 그대로 사용) ...
        # (지면 관계상 생략, 이전에 드린 코드를 넣으세요)
        if not os.path.exists(filepath): return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
            if "}{" in content: content = content.replace("}{", "}\n{")
            data = []
            for line in content.splitlines():
                if line.strip(): 
                    try: data.append(json.loads(line))
                    except: pass
            return data
        except: return []

    def _init_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- 색상 팔레트 ---
        TOP_BG = '#2d2d2d'     
        BOTTOM_BG = '#1e1e1e'  
        TEXT_COLOR = 'white'
        
        self.root.configure(bg=TOP_BG)
        
        style.configure('.', background=BOTTOM_BG, foreground=TEXT_COLOR)
        style.configure('TFrame', background=BOTTOM_BG)
        style.configure('Header.TFrame', background=TOP_BG)
        style.configure('Header.TLabel', background=TOP_BG, foreground=TEXT_COLOR)

        # ==========================================================
        # [수정] 스크롤바: 두께 문제 해결 (arrowsize=16 복구)
        # ==========================================================
        style.layout('Vertical.TScrollbar', 
                     [('Vertical.Scrollbar.trough',
                       {'children': [('Vertical.Scrollbar.thumb', 
                                      {'expand': '1', 'sticky': 'nswe'})],
                        'sticky': 'nswe'})]) # 좌우(we)로 꽉 채우기
        
        # 썸(Thumb) 색상 정의
        THUMB_COLOR = '#555555'
        THUMB_BACKGROUND_COLOR = "#444444"
        style.configure('Vertical.TScrollbar',
                        orient='vertical',
                        width=16,              # 설정한 너비
                        arrowsize=16,          # [해결책] 이 값을 너비와 맞춰야 공간이 확보됩니다!
                        gripcount=0,
                        troughcolor=BOTTOM_BG, 
                        background=THUMB_COLOR,
                        relief='flat',
                        borderwidth=0,
                        bordercolor=BOTTOM_BG,
                        lightcolor=THUMB_COLOR,
                        darkcolor=THUMB_COLOR,
                        troughborderwidth=0, 
                        troughrelief='flat')

        style.map('Vertical.TScrollbar',
                  background=[('pressed', '#777777'), ('active', '#666666')],
                  troughcolor=[('active', BOTTOM_BG)],
                  bordercolor=[('active', BOTTOM_BG)],
                  lightcolor=[('active', BOTTOM_BG)],
                  darkcolor=[('active', BOTTOM_BG)])

        # ----------------------------------------------------------
        # 탭 스타일 (기존 유지)
        # ----------------------------------------------------------
        style.configure('TNotebook', 
                        background=TOP_BG,  
                        borderwidth=0, 
                        tabmargins=[0, 0, 0, 0],
                        lightcolor=TOP_BG, 
                        darkcolor=TOP_BG, 
                        bordercolor=TOP_BG)

        TAB_PADDING = [30, 10] 

        style.configure('TNotebook.Tab', 
                        background=TOP_BG,      
                        foreground='#888888',   
                        padding=TAB_PADDING,
                        font=('Arial', 12, 'bold'),
                        borderwidth=0,            
                        focuscolor=TOP_BG,      
                        bordercolor=TOP_BG,    
                        lightcolor=TOP_BG,     
                        darkcolor=TOP_BG)      

        style.map('TNotebook.Tab', 
                  background=[('selected', BOTTOM_BG)], 
                  foreground=[('selected', 'white')],
                  lightcolor=[('selected', BOTTOM_BG)],
                  darkcolor=[('selected', BOTTOM_BG)],
                  bordercolor=[('selected', BOTTOM_BG)],
                  padding=[('selected', TAB_PADDING)], 
                  expand=[('selected', [0, 0, 0, 0])]) 

        # 콤보박스
        style.configure('TCombobox', fieldbackground='#333333', background='#333333', foreground='white', arrowcolor='white', borderwidth=0)
        style.map('TCombobox', fieldbackground=[('readonly', '#333333')], selectbackground=[('readonly', '#333333')], selectforeground=[('readonly', 'white')])

        # ----------------------------------------------------------
        # UI 배치
        # ----------------------------------------------------------
        
        # 헤더
        header_frame = ttk.Frame(self.root, padding=20, style='Header.TFrame')
        header_frame.pack(side='top', fill='x')
        
        lbl_title = ttk.Label(header_frame, text="🍷 Vivino Flavor Studio", font=('Helvetica', 24, 'bold'), style='Header.TLabel')
        lbl_title.pack(side='left')
        
        count = len(self.wine_metadata) if hasattr(self, 'wine_metadata') else 0
        lbl_count = ttk.Label(header_frame, text=f"Total Wines: {count}", font=('Arial', 12), foreground='#666666', style='Header.TLabel')
        lbl_count.pack(side='right', anchor='s')

        # [수정] 탭 컨테이너 여백 추가
        notebook = ttk.Notebook(self.root)
        
        # padx=20, pady=20: 탭 창 주변에 여백을 줌 -> 뒤쪽의 TOP_BG 색상이 보임
        notebook.pack(side='top', fill='both', expand=True, padx=20, pady=(0, 20)) 

        # 탭 추가
        tab1 = SearchTab(notebook, self.wine_metadata, self.analyzer)
        notebook.add(tab1, text="Search Wines") 
        
        # [수정됨] 탭 2: 카테고리 (My Cellar -> Category)
        # 기존: tab2 = ttk.Frame(notebook); notebook.add(tab2, text="My Cellar")
        # 변경:
        tab2 = CategoryTab(notebook, self.wine_metadata, self.analyzer)
        notebook.add(tab2, text="Category Explorer")

        tab3 = AnalyticsTab(notebook, self.wine_metadata, self.analyzer)
        notebook.add(tab3, text="Analytics (Similarity)")  # <--- 여기!

        tab4 = ttk.Frame(notebook); notebook.add(tab4, text="Settings")

if __name__ == "__main__":
    root = tk.Tk()
    app = WineApp(root)
    root.mainloop()