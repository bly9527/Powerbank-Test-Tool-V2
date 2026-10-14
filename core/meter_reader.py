"""Read meter time and capacity from one evidence frame, always storing Ah."""
from core.ocr_engine import OCREngine


def read_meter_fields(time_image, capacity_image, engine, cap_unit='Ah', single_line=False):
    images = {'time': time_image, 'cap': capacity_image}
    time_reader = capacity_reader = engine
    if single_line:
        # The user explicitly confirms each ROI contains a single meter row.
        # Avoid text detection splitting tiny decimals/colons into separate boxes.
        readers = []
        for key in ('time', 'cap'):
            result, _ = engine.ocr(images[key], use_det=False, use_cls=False)
            boxes = []
            if result and len(result) == 1 and float(result[0][1]) >= .9:
                text, confidence = result[0][:2]
                if key == 'cap' and cap_unit == 'mAh' and 'ah' not in text.lower():
                    text += ' mAh'
                boxes = [[[[0,0],[100,0],[100,20],[0,20]], text, confidence]]
            reader = OCREngine.__new__(OCREngine)
            reader.preprocess_soc_crop = lambda image: image
            reader.ocr = lambda image, boxes=boxes: (boxes, None)
            readers.append(reader)
        time_reader, capacity_reader = readers
    t, obj, tc = time_reader.extract_time(time_image)
    c, raw, cc = capacity_reader.extract_capacity(capacity_image, target_unit='Ah')
    if not single_line and cap_unit == 'mAh' and c is not None and 'ah' not in raw.lower():
        # The selected unit also applies when the ROI excludes the printed unit.
        c, _, _ = capacity_reader.extract_capacity(capacity_image, target_unit='raw')
        c = round(c / 1000., 4) if c is not None else None
    return {'time_str': t or '--', 'time_obj': obj, 'capacity': c,
            'capacity_raw': raw, 'time_confidence': tc, 'capacity_confidence': cc}
