## 项目结构
    - commons: 工具类
    - models: 模型
    - pdf_img_classification.py: 训练图像分类
    - pdf_txt_classification.py: 训练文本分类
## 依赖
    pip install jieba==0.39
    pip install tensorflow==1.6.0
    pip install fitz PyMuPDF pillow
    pip install pdf2image==1.9.0
    conda install -c conda-forge poppler
    pip install pytesseract==0.3.0
    pip install pdfplumber==0.5.23
## 安装tesseract-ocr
    安装包路径:\\cnshf-fs01\TFS_Automation_CN\95_Deep_Learning_Classification\tesseract-ocr-setup-4.00.00dev.exe
## 训练文本分类
    执行以下命令即可
    python pdf_txt_classification.py --train_folder_name test_train --pdf_dir \\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\online_pdf
    train_folder_name: 新建一个目录用于保存训练生成的数据。不传该参数时默认创建一个当前日期目录
    pdf_dir: 已标注好的pdf所在路径。不传该参数时默认为 \\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\online_pdf
