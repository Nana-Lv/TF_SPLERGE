#-*-coding:utf-8-*-
import os
import fitz
import json
from xml.dom.minidom import parseString
import cv2


class PDF2PNG:
    def __init__(self):
        return
    
    @classmethod
    def pdf2png(cls, pdf_path, save_dir):
        pdf_doc = fitz.open(pdf_path)
        for pg in range(pdf_doc.pageCount):
            page = pdf_doc[pg]
            zoom_x, zoom_y = 1, 1   # 尺寸的缩放系数，维持不变就可以和标注信息对应
            mat = fitz.Matrix(zoom_x, zoom_y).preRotate(int(0))
            pix = page.getPixmap(matrix=mat, alpha=False)
            image_path = os.path.join(save_dir, str(pg) + '.png')
            pix.writePNG(image_path)
        return


class ProcICDAR2013:
    def __init__(self, dataset_path, png_path, dst_path):
        """
        You will get the train data in the dst_path.
        :param dataset_path:
        :param png_path:
        :param dst_path:
        """
        self.dataset_path = dataset_path
        self.png_path = png_path
        self.dst_path = dst_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.triple = self.get_triple(dataset_path)     # pdf, reg, str
        self.quadruple = self.cvt_pdf_to_png(png_path)  # pdf, reg, str, png
        self.proc_all_pdf()
        return
    
    def get_triple(self, dataset_path):
        triple = []
        for sub_dir, mid_dir, files in os.walk(dataset_path):
            for fi in files:
                sprt = fi.split('.')
                if len(sprt) != 2:
                    continue
                name, ext = sprt
                if ext != 'pdf':
                    continue
                p = os.path.join(sub_dir, fi)
                r = p.replace('.pdf', '-reg.xml')
                s = p.replace('.pdf', '-str.xml')
                triple.append([p, r, s])
        return triple
    
    def cvt_pdf_to_png(self, png_path):
        for i, (pdf, *_) in enumerate(self.triple):
            temp = pdf[len(dataset_path):]
            pdf_name = temp.split('/')[-1]
            _set = temp[: - len(pdf_name)]
            _save_dir = os.path.join(png_path, _set.strip('/'), pdf_name.split('.')[0].strip('/'))
            if not os.path.exists(_save_dir):
                os.makedirs(_save_dir)
            PDF2PNG.pdf2png(pdf, _save_dir)
            self.triple[i].append(_save_dir)
        return self.triple

    def proc_single_pdf(self, reg_path, str_path, png_dir):
        def cvt_merge_info(idx):
            if idx:
                return int(idx)
            return -1
        
        # reg file
        with open(reg_path, 'rb') as f:
            content = f.read()
        _reg = parseString(bytes.decode(content, encoding='utf-8'))
        tables_reg = _reg.getElementsByTagName('table')
        # str file
        with open(str_path, 'rb') as f:
            content = f.read()
        _str = parseString(bytes.decode(content, encoding='utf-8'))
        tables_str = _str.getElementsByTagName('table')
        # file linkage
        for i, (table_reg, table_str) in enumerate(zip(tables_reg, tables_str)):
            region = table_reg.getElementsByTagName('region')[0]
            pg = region.getAttribute('page')
            # get png img info
            png_path = os.path.join(png_dir, str(int(pg) - 1) + '.png')
            img = cv2.imread(png_path)
            h, w, c = img.shape
            # get bbox for one table
            bbox = region.getElementsByTagName('bounding-box')[0]
            tx1, ty1, tx2, ty2 = map(int, [bbox.getAttribute('x1'), bbox.getAttribute('y1'),
                                           bbox.getAttribute('x2'), bbox.getAttribute('y2')])
            tx1, ty1, tx2, ty2 = map(int, [tx1, h - ty2, tx2, h - ty1])
            # gen img path for one table and save the table png
            img_path = os.path.join(self.dst_path,
                                    reg_path[len(self.dataset_path): -len('-reg.xml')].strip('/').replace('/', '-')
                                    + '-' + str(i) + '.png')
            table_img = img[ty1: ty2, tx1: tx2, :]
            cv2.imwrite(img_path, table_img)
            # proc cells info for the table above
            cells = table_str.getElementsByTagName('cell')
            json_path = img_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                for cell in cells:
                    # get merge info for one cell
                    sr, er, sc, ec = cell.getAttribute('start-row'), cell.getAttribute('end-row'), \
                                     cell.getAttribute('start-col'), cell.getAttribute('end-col')
                    sr, er, sc, ec = [x for x in map(cvt_merge_info, [sr, er, sc, ec])]
                    er = -1 if er == sr else er
                    ec = -1 if ec == sc else ec
                    merge_info = [sr, er, sc, ec]
                    # get bbox for one cell
                    bbox = cell.getElementsByTagName('bounding-box')[0]
                    cx1, cy1, cx2, cy2 = map(int, [bbox.getAttribute('x1'), bbox.getAttribute('y1'),
                                                   bbox.getAttribute('x2'), bbox.getAttribute('y2')])
                    bbox_info = [x for x in map(int, [cx1 - tx1, h - cy2 - ty1, cx2 - tx1, h - cy1 - ty1])]
                    table_grount_truth = {'merge_info': merge_info, 'bbox_info': bbox_info}
                    f.write(json.dumps(table_grount_truth, ensure_ascii=False))
                    f.write('\n')
                    x1, y1, x2, y2 = bbox_info
                    cv2.rectangle(table_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        return
    
    def proc_all_pdf(self):
        for pdf_path, reg_path, str_path, png_dir in self.quadruple:
            self.proc_single_pdf(reg_path, str_path, png_dir)
        return


def plot_rect_on_table_img(data_path, test_path):
    """
    You can check the rects on img from the test_path.
    :param data_path:
    :param test_path:
    :return:
    """
    files = os.listdir(data_path)
    for fi in files:
        if fi.split('.')[-1] != 'png':
            continue
        js = fi.replace('.png', '.json')
        img = cv2.imread(os.path.join(data_path, fi))
        with open(os.path.join(data_path, js), 'r') as f:
            cells = f.readlines()
            for cell in cells:
                cell = eval(cell.strip('\n'))
                x1, y1, x2, y2 = cell['bbox_info']
                cv2.rectangle(img, (x1, y1), (x2, y2), thickness=1, color=(0, 0, 255))
        dst_path = os.path.join(test_path, fi)
        cv2.imwrite(dst_path, img)
    return


if __name__ == '__main__':
    dataset_path = '/home/yx-lxd/PycharmProjects/TF_SPLERGE/ICDAR2013_table_dataset'
    png_path ='/home/yx-lxd/PycharmProjects/TF_SPLERGE/PNGs'
    dst_path ='/home/yx-lxd/PycharmProjects/TF_SPLERGE/splerge_data'
    PI = ProcICDAR2013(dataset_path, png_path, dst_path)
    test_path = '/home/yx-lxd/PycharmProjects/TF_SPLERGE/check_img'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    plot_rect_on_table_img(dst_path, test_path)
    
