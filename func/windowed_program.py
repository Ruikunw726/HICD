import sys
sys.path.append('M:/')

import rasterio
import argparse
from rasterio.plot import reshape_as_image
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from MambaCD.changedetection.script import infer_MambaBDA
import MambaCD.changedetection.datasets.imutils as imutils
from MambaCD.changedetection.datasets.make_data_loader import img_loader

import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认中文字体
mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def get_args():
    parser = argparse.ArgumentParser(description="Inference on xBD dataset")
    parser.add_argument('--cfg', type=str, default=r'M:/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--test_dataset_path', type=str, default=r'M:/MambaCD/test')
    parser.add_argument('--test_data_list_path', type=str, default=r'M:/MambaCD/test.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--model_type', type=str, default='MambaBDA_Tiny')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str,default=r'M:/MambaCD/changedetection/saved_models/data_test_new_600_model.pth')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    
    return args

# 加载模型 - 
def load_model():
    
    args = get_args()
    
    trainer = infer_MambaBDA.Trainer(args)
    
    model = trainer.deep_model
    
    return model


# 变化检测预测函数
def predict_change(model, img1_path, img2_path):
    # 预处理图像
    pre_img = img_loader(img1_path)
    post_img = img_loader(img2_path)

    
    pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
    pre_img = np.transpose(pre_img, (2, 0, 1))

    post_img = imutils.normalize_img(post_img)  # imagenet normalization
    post_img = np.transpose(post_img, (2, 0, 1))
    
    
    img1 = torch.from_numpy(pre_img).float().cuda()
    img2 = torch.from_numpy(post_img).float().cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    
    # 使用模型进行预测
    with torch.no_grad():
        output_loc, output_clf = model(img1, img2)
        
    
    # 为可视化创建伪彩色图
    output_clf = output_clf.data.cpu().numpy()
    output_clf = np.argmax(output_clf, axis=1)
    
    output_loc = output_loc.data.cpu().numpy()
    output_loc = np.argmax(output_loc, axis=1)
        
    output_clf = np.squeeze(output_clf).astype(np.uint8)
    output_loc = np.squeeze(output_loc)
    
    output_clf[output_loc == 0] = 0

    output_clf = infer_MambaBDA.map_labels_to_colors(np.squeeze(output_clf), 
                                                     ori_label_value_dict=infer_MambaBDA.ori_label_value_dict1, 
                                                     target_label_value_dict=infer_MambaBDA.target_label_value_dict1)
    
    return output_loc ,output_clf

# 主应用类
class ChangeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("遥感影像战争损坏评估系统")
        self.root.geometry("1600x800")
        
        # 加载预训练模型
        self.model = load_model()
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 影像1按钮
        self.btn_img1 = ttk.Button(self.control_frame, text="加载影像1", command=self.load_img1)
        self.btn_img1.pack(padx=10, pady=10, fill=tk.X)
        
        # 影像2按钮
        self.btn_img2 = ttk.Button(self.control_frame, text="加载影像2", command=self.load_img2)
        self.btn_img2.pack(padx=10, pady=10, fill=tk.X)
        
        # 执行分析按钮
        self.btn_analyze = ttk.Button(self.control_frame, text="执行变化检测", 
                                     command=self.analyze_changes, state=tk.DISABLED)
        self.btn_analyze.pack(padx=10, pady=10, fill=tk.X)
        
        # 结果面板
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        
        # 图像显示区域
        self.display_frame = ttk.LabelFrame(self.main_frame, text="影像显示")
        self.display_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # 创建双列框架用于并排显示两张图像
        self.image_grid_frame = ttk.Frame(self.display_frame)
        self.image_grid_frame.pack(fill='both', expand=True)
        
        # 状态栏
        self.status_bar = ttk.Label(root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化图像变量
        self.img1_path = None
        self.img2_path = None
        self.img1_display = None
        self.img2_display = None
        self.result_display = None
        
        # 配置网格权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # 配置图像网格的列权重
        self.image_grid_frame.columnconfigure(0, weight=1)
        self.image_grid_frame.columnconfigure(1, weight=1)
        
        # 创建影像1展示区域
        self.img1_frame = ttk.LabelFrame(self.image_grid_frame, text="第一景影像")
        self.img1_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.img1_label = ttk.Label(self.img1_frame)
        self.img1_label.pack(padx=5, pady=5)
        
        # 创建影像2展示区域
        self.img2_frame = ttk.LabelFrame(self.image_grid_frame, text="第二景影像")
        self.img2_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.img2_label = ttk.Label(self.img2_frame)
        self.img2_label.pack(padx=5, pady=5)
    
    
    def load_img1(self):
        self.img1_path = filedialog.askopenfilename(
            title="选择第一期遥感影像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")]
        )
        if self.img1_path:
            self.show_image(self.img1_path, "影像1加载成功", self.img1_label, "img1_photo")
            self.check_analysis_ready()
    
    def load_img2(self):
        self.img2_path = filedialog.askopenfilename(
            title="选择第二期遥感影像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")]
        )
        if self.img2_path:
            self.show_image(self.img2_path, "影像2加载成功", self.img2_label, "img2_photo")
            self.check_analysis_ready()
    
    def check_analysis_ready(self):
        if self.img1_path and self.img2_path:
            self.btn_analyze.config(state=tk.NORMAL)
    
    def show_image(self, path, message, target_label, photo_var):
        try:
            img = Image.open(path)
            img.thumbnail((180, 180))  # 调整大小以适应界面
            
            # 保存为PhotoImage对象并显示
            photo = ImageTk.PhotoImage(img)
            
            # 保存引用，防止被垃圾回收
            setattr(self, photo_var, photo)
            
            # 更新标签的图像
            target_label.configure(image=photo)
            
            self.status_bar.config(text=message)
        except Exception as e:
            self.status_bar.config(text=f"错误: {str(e)}")
    
    def analyze_changes(self):
        if not self.img1_path or not self.img2_path:
            self.status_bar.config(text="请先加载两期遥感影像")
            return
        
        self.status_bar.config(text="正在分析变化...")
        self.root.update()
        
        try:

            img1 = img_loader(self.img1_path)
            img2 = img_loader(self.img2_path)
            # 使用模型进行预测
            loc_map, change_map = predict_change(
                self.model, self.img1_path, self.img2_path
            )
            
            # 显示结果
            self.show_results(change_map)
            self.status_bar.config(text="变化检测完成")
            
        except Exception as e:
            self.status_bar.config(text=f"分析错误: {str(e)}")
    
    def show_results(self, result_img):
                
        # 清除旧的结果
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # 创建图表
        fig = plt.Figure(figsize=(8, 6))
        
        # 图像1
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(result_img)
        ax1.set_title("变化检测结果 (红色为损坏，绿色为未损坏)")
        ax1.axis('off')
        
        fig.tight_layout()
        
        # 嵌入Tkinter画布
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = ChangeDetectionApp(root)
    root.mainloop()