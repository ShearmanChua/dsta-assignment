import io
import json
import pathlib
import pydash
import numpy as np
import PIL
from PIL import Image, ExifTags
from fitz import fitz
from assignment_1.assignment_1_1.helper_convert import extract_page_blocks, normalize
from assignment_1.assignment_1_1.tesseract_ocr import TesseractOCR, TesseractResultTransformer

def image_to_pdf(image_path):
    with open(image_path, 'rb') as file:
        bin_in = file.read()
        bin_in = io.BytesIO(bin_in)

    try:
        pil_image = Image.open(bin_in)
    except PIL.UnidentifiedImageError:
        raise

    if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
        pil_image = pil_image.convert('RGB')

    for orientation_key in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation_key] == 'Orientation':
            if hasattr(pil_image, '_getexif'):
                img_exif = pil_image._getexif()
                if hasattr(img_exif, 'items'):
                    exif = dict(img_exif.items())
                    orientation = exif.get(orientation_key)
                    if orientation == 1:
                        # Normal image - nothing to do
                        pass
                    elif orientation == 2:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        pil_image = pil_image.rotate(180, expand=True)
                    elif orientation == 4:
                        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
                    elif orientation == 6:
                        pil_image = pil_image.rotate(-90, expand=True)
                    elif orientation == 7:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                    elif orientation == 8:
                        pil_image = pil_image.rotate(90, expand=True)
            break

    pdf_fp_out = io.BytesIO()
    pil_image.save(pdf_fp_out, "PDF", creationDate=None, modDate=None)
    pdf_fp_out.seek(0)

    return pdf_fp_out

def convert_one_page_from_pdf_page(pdf_doc, page_id):
    pdf_page = pdf_doc[page_id]
    rgb_array = pdf_render_page(pdf_page)
    img_pil = Image.fromarray(rgb_array)
    response = TesseractOCR().detect_text_in_image_sync(img_pil)
    rt = TesseractResultTransformer(np.array(rgb_array))
    ocr_data = {page_id + 1: {"responses": response}}
    if hasattr(rt, 'add_normalized_bbox'):
        rt.add_normalized_bbox(ocr_data)
    filename = "-"
    # 后处理
    ocr_data_trans = rt.transform(filename, filename, ocr_data)
    idp_page = {
        **ocr_data_trans["pages"][0],
        "bbox": [0, 0, rgb_array.shape[1], rgb_array.shape[0]],
    }
    return idp_page


def pdf_render_page(page: fitz.Page):
    pix = page.get_pixmap()
    img_Image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    canvas = np.array(img_Image)
    return canvas


def ocr_transform_to_idp_format(pdf_bin, selected_pages=None):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    pages = []
    map_type_to_names = {
        0: 'text',
        1: 'image'
    }
    page_count = doc.page_count
    for page_id in range(page_count):
        # Check if the page_id + 1 (1-based index) is in the selected_pages list
        if selected_pages and (page_id + 1) not in selected_pages:
            continue

        page_blocks = extract_page_blocks(doc.load_page(page_id))
        groups = pydash.group_by([b['bbox'] + (b['type'],) for b in page_blocks], 4)
        blocks_by_type = {map_type_to_names[k]: len(v) for k, v in groups.items()}
        if blocks_by_type.get('image', 0) <= 0:
            continue
        pages.append(convert_one_page_from_pdf_page(pdf_doc=doc, page_id=page_id))
    return {"pages": pages}


def convert_scanned_pdf(pdf_bin, selected_pages=None):
    ocr_convert_result = ocr_transform_to_idp_format(pdf_bin, selected_pages)
    text_result = normalize(ocr_convert_result)
    return text_result


if __name__ == "__main__":
    dataset_dir = pathlib.Path(__file__).parents[1]
    image_path = 'assignment_1_1/data/down.jpg'
    pdf_blob = image_to_pdf(image_path)
    with open('assignment_1_1/data/down.pdf', 'wb') as pdf_file:
        pdf_file.write(pdf_blob.getvalue())
    input_path = 'assignment_1_1/data/down.pdf'
    file_path = dataset_dir / input_path
    save_path = 'assignment_1_1/data/down'
    test_file = dataset_dir / f'{save_path}.json'

    # Specify the pages to be converted (1-based index)
    # selected_pages = [3, 5, 7]

    pdf_bin = file_path.read_bytes()
    result = convert_scanned_pdf(pdf_bin)
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)