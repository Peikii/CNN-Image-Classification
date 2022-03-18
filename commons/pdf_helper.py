import os
import re
import fitz # pdf-->img
import pdf2image # pdf-->img
import pytesseract # img-->txt
import pdfplumber # pdf-->txt
from PIL import Image
from io import BytesIO

# 配置环境变量
if os.name == 'nt': # windows系统添加tesseract环境变量。
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

def pdf2img_fitz(pdf_path, page_num=None, size=(None, None), is_padding=False):
    """
    用fitz将pdf转图片
    :param pdf_path: pdf路径
    :param page_num: 指定页；传None则获取全部页、
                            传整数则获取指定页、
                            传字符串，如‘1-3’则获取1-3页（下标从0开始）
    :param size: (w, h), resize图像尺寸(单页)； 若某个维度为None，则继承图像原有尺寸
    :param is_padding: 页数不够时是否拼接空白页；只有获取指定几页时才有效
    :return: Image类型图像
    """
    try:
        if (isinstance(page_num, str) and "-" in page_num):
            start, end = page_num.split("-")
            start, end = int(start), int(end)
            total_page = end - start + 1
        else:
            start, end = None, None
            total_page = None

        with fitz.open(pdf_path) as pdfDoc:
            w, h = 0, 0
            img_list = []
            if page_num is None or (isinstance(page_num, str) and "-" in page_num):
                for i in range(pdfDoc.pageCount):
                    if start is not None and end is not None:
                        if not (i >= start and i <= end):
                            continue
                    page = pdfDoc[i]
                    pix = page.getPixmap(alpha=False)
                    img_barray = pix.getImageData() # 获取图像字节流

                    img = Image.open(BytesIO(img_barray)) # 转图像对象
                    size = (size[0] if size[0] is not None else pix.width, size[1] if size[1] is not None else pix.height)
                    img = img.resize(size) # 调整尺寸

                    w_i, h_i = img.size # pix.width, pix.height
                    if w_i > w:
                        w = w_i
                    h = h + h_i

                    # img.save('images_%s.png' % i)
                    img_list.append(img)
                return img_list
                # 拼图
                #if is_padding:
                #    if len(img_list) < total_page:
                #        h = h + size[1] * (total_page - len(img_list))
                #result = Image.new("RGB", (w, h), color=(255, 255, 255))
                #box = (0, 0) # 起始坐标con
                #for i, img in enumerate(img_list):
                #    result.paste(img, box=box)
                #    box = (0, box[1] + img.size[1])
                # return result
            elif isinstance(page_num, int):
                page = pdfDoc[page_num]
                pix = page.getPixmap(alpha=False)
                img_barray = pix.getImageData()  # 获取图像字节流

                img = Image.open(BytesIO(img_barray))  # 转图像对象
                size = (size[0] if size[0] is not None else pix.width, size[1] if size[1] is not None else pix.height)
                img = img.resize(size)  # 调整尺寸

                return img
    except Exception as e:
        raise e

def pdf2txt_pdfplumber(pdf_path=None, page_num=None):
    """
    用pdfplumber将pdf转文字。
    :param pdf_path:
    :param page_num: None(取全部)
    :return:
    """
    try:
        txt = []
        # with open(pdf_path, 'rb') as fp:
        with pdfplumber.open(pdf_path) as pdf:
            if isinstance(page_num, int):
                page_txt = pdf.pages[page_num].extract_text()
                if page_txt is not None:
                    txt.append(page_txt)
            elif isinstance(page_num, str):
                start, end = page_num.split("-")
                start, end = int(start)+1, int(end)+1
                for page in pdf.pages:
                    curr_page = page.page_number # 下标从1开始
                    if curr_page >= start and curr_page <= end:
                        page_txt = page.extract_text()
                        if page_txt is not None:
                            txt.append(page_txt)
            elif page_num is None:
                for page in pdf.pages:
                    page_txt = page.extract_text()
                    if page_txt is not None:
                        txt.append(page_txt)
        if txt:
            return re.sub(r'\s+', ' ', "".join(txt).strip())
        else:
            return None
    except Exception as e:
        raise e

def pdf2txt_ocr(path, page_num=None, split_img=False):
    """
    获取PDF内容
    :param path: PDF路径
    :param page_num: 为None，则获取全部页；为 0， 1,2，3.。。，则获取指定页；为 1-2 获取指定区间
    :return:
    """
    try:
        if page_num is None: # 获取全部页内容
            # img, page_size, w_h = pdf2img(path=path)
            # img = img.crop((0, 0, w_h[0], 500))
            # img.show()
            content = ""
            img_list = pdfperpage2img(path=path)
            for img in img_list:
                content_i = img2txt(img)
                if content_i is not None:
                    content = content + content_i
        elif isinstance(page_num, str) and "-" in page_num:
            start, end = page_num.split("-")
            img_list = pdfmutipage2img(path, int(start), int(end))
            content = ""
            for img in img_list:
                w, h = img.size
                if split_img:
                    img = img.crop((0, 0, w, h/2)) # 取一半位置
                content_i = img2txt(img)
                if content_i is not None:
                    content = content + content_i
        elif isinstance(page_num, int): # 获取指定页内容
            img = pdfpage2img(path=path, page_index=page_num)
            content = img2txt(img)
        else:
            pass
        # 合并空格
        content = re.sub(r'\s+', ' ', content.strip())
        return content
    except Exception as e:
        raise e


#################################################

def pdfpage2img(path, page_index=0):
    """
    pdf指定页转图片
    :param path:
    :param page_index:
    :return:
    """
    try:
        imges = pdf2image.convert_from_path(pdf_path=path)
        pagei = imges[page_index]
        w_i, h_i = pagei.size
        result = Image.new("RGB", (w_i, h_i))
        result.paste(pagei, box=(0, 0))
    except Exception as e:
        raise e
    return result

def pdfperpage2img(path):
    """
    pdf每一页转一个图片
    :param path:
    :return: 图片数组
    """
    try:
        img_list = []
        imges = pdf2image.convert_from_path(pdf_path=path)
        for index, img in enumerate(imges):
            w_i, h_i = img.size
            result = Image.new("RGB", (w_i, h_i))
            result.paste(img, box=(0, 0))
            img_list.append(result)
        return img_list
    except Exception as e:
        raise e


def pdfmutipage2img(path, start, end):
    """
    pdf指定几页转img
    如果页数不够，获取最大页
    :param path:
    :param start:开始下标
    :param end:结束下标
    :return: 图片数组
    """
    try:
        img_list = []
        imges = pdf2image.convert_from_path(pdf_path=path)
        for index, img in enumerate(imges):
            # print(index)
            if index >= start and index <=end:
                w_i, h_i = img.size
                result = Image.new("RGB", (w_i, h_i))
                result.paste(img, box=(0, 0))
                img_list.append(result)
        return img_list
    except Exception as e:
        raise e

def img2txt(img, config="--psm 6"):
    """
    ocr
    :param img: 路径或Image对象
    :param config: "--psm 7" 单个文本行
                    "--psm 6" 假设单个统一的文本块
    :return:
    """
    if isinstance(img, str):
        img = Image.open(img)
    # else:
    #     img = Image.fromarray(img)
    # print(pytesseract.get_tesseract_version())
    content = pytesseract.image_to_string(img, config=config)
    return content

# if __name__ == '__main__':
#     import fitz
#     print(fitz.version)