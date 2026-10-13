"""Independent, explicitly labelled image libraries for segmented displays."""
import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.roi_geometry import crop_roi

SEGMENT_MODE = 4  # Existing OCR indices 0..3 must keep their original meaning.


def data_home():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'PowerbankTestTool' / 'segment_samples'


def library_paths(value):
    values = value if isinstance(value, (list, tuple)) else ([value] if value else [])
    return list(dict.fromkeys(str(Path(p)) for p in values if p))


def template_catalog(extra_paths=()):
    """Discover legacy libraries without moving or rewriting their labels."""
    entries = read_json(data_home() / 'templates.json', [])
    known = {str(Path(e['path']).resolve()).casefold() for e in entries}
    candidates = list(extra_paths)
    if data_home().exists():
        candidates += [p for p in data_home().iterdir() if p.is_dir() and (p/'captures').exists()]
    for path in candidates:
        path = Path(path).resolve()
        if str(path).casefold() not in known:
            entries.append({'name': path.name, 'path': str(path)})
            known.add(str(path).casefold())
    return entries


def register_template(name, path=None):
    name = name.strip()
    if not name:
        raise ValueError('请输入模板名称')
    entries = template_catalog()
    path = Path(path).resolve() if path else data_home() / ('template_' + uuid4().hex)
    path.mkdir(parents=True, exist_ok=True)
    target = next((e for e in entries if Path(e['path']).resolve() == path.resolve()), None)
    if target:
        target['name'] = name
    else:
        entries.append({'name': name, 'path': str(path)})
    write_json(data_home() / 'templates.json', entries)
    return str(path)


def import_dataset(source, destination):
    """Append copies with fresh batch IDs; keep existing labels and originals."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('来源数据集与目标模板相同')
    sessions = capture_sessions(source)
    if not sessions:
        raise ValueError('此文件夹没有采集批次，请选择含 captures 的样本库')
    source_regions = load_ignore_regions(source)
    target_regions = load_ignore_regions(destination)
    configured = (destination / 'recognition.json').exists() or bool(capture_sessions(destination))
    if configured and source_regions != target_regions:
        raise ValueError('两个数据集的忽略区域不同，请单独导入为新模板，避免遮住数字。')
    if not configured:
        save_ignore_regions(destination, source_regions)
    original_labels = read_json(source / 'labels.json', {})
    total = 0
    for session in reversed(sessions):
        old_id, new_id = session['id'], uuid4().hex
        target = destination / 'captures' / new_id
        target.mkdir(parents=True)
        copied = []
        labels = read_json(destination / 'labels.json', {})
        for item in session['samples']:
            relative = Path('captures') / old_id / item['image']
            origin = (source / relative).resolve()
            if not origin.is_relative_to((source/'captures'/old_id).resolve()):
                raise ValueError('数据集中含有无效的图片路径')
            name = f'{len(copied):07d}.png'
            shutil.copyfile(origin, target / name)
            copied.append(dict(item, image=name))
            if relative.as_posix() in original_labels:
                labels[(Path('captures') / new_id / name).as_posix()] = original_labels[relative.as_posix()]
        session.update(samples=copied, id=new_id, created_at=time.time(), imported_from=str(source))
        write_json(target / 'manifest.json', session)
        write_json(destination / 'labels.json', labels)
        total += len(copied)
    return total


def export_template_package(root, output_path, name):
    """Create a portable package containing confirmed labels and their images."""
    root = Path(root).resolve()
    labels = read_json(root / 'labels.json', {})
    kept = {key: value for key, value in labels.items() if type(value) is int and 0 <= value <= 100}
    if not kept:
        raise ValueError('此模板还没有已确认的数字样本')
    regions = load_ignore_regions(root)
    count = 0
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        # Older software must not silently ignore masking rules in a shared template.
        package.writestr('template.json', json.dumps({'version': 2 if regions else 1, 'name': name}, ensure_ascii=False))
        package.writestr('recognition.json', json.dumps(
            {'version': 1, 'ignore_regions': regions}))
        for session in capture_sessions(root):
            copied = []
            for item in session['samples']:
                relative = Path('captures') / session['id'] / item['image']
                if relative.as_posix() not in kept:
                    continue
                path = (root / relative).resolve()
                if not path.is_relative_to((root / 'captures' / session['id']).resolve()):
                    raise ValueError('模板中包含不安全的图片路径')
                package.write(path, relative.as_posix())
                copied.append(item)
                count += 1
            if copied:
                manifest = dict(session, samples=copied, source=Path(session.get('source', '')).name)
                manifest.pop('id', None)
                package.writestr(f'captures/{session["id"]}/manifest.json',
                    json.dumps(manifest, ensure_ascii=False))
        package.writestr('labels.json', json.dumps(kept, ensure_ascii=False))
    return count


def import_template_package(package_path):
    """Validate and import a shared zip as a new named template."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(package_path) as package:
            names = package.namelist()
            if 'template.json' not in names or 'labels.json' not in names:
                raise ValueError('不是有效的数码管模板包')
            for info in package.infolist():
                path = Path(info.filename)
                if path.is_absolute() or '..' in path.parts or path.suffix.lower() not in ('.json', '.png'):
                    raise ValueError('模板包包含不安全的文件路径')
                if info.file_size > 100 * 1024 * 1024:
                    raise ValueError('模板包中的文件过大')
                destination = (root / path).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError('模板包包含不安全的文件路径')
                if info.is_dir():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
        info = read_json(root / 'template.json', {})
        if info.get('version') not in (1, 2) or not str(info.get('name', '')).strip():
            raise ValueError('模板包版本或名称无效')
        if info['version'] == 2 and not (root / 'recognition.json').is_file():
            raise ValueError('模板包缺少忽略区域配置')
        labels = read_json(root / 'labels.json', {})
        if not any(type(value) is int and 0 <= value <= 100 for value in labels.values()):
            raise ValueError('模板包中没有已确认的数字样本')
        load_ignore_regions(root)  # Validate before registering or copying anything.
        destination = register_template(info['name'])
        try:
            count = import_dataset(root, destination)
        except Exception:
            # Do not remove a registered destination when a copy fails: its
            # contents may still be useful for inspection or recovery.
            raise
        return destination, count


def clean_unlabelled_capture_batches(root, batch_ids):
    """Only prune unlabeled images from batches captured in this dialog session."""
    root = Path(root).resolve()
    labels = read_json(root / 'labels.json', {})
    removed = 0
    for batch_id in set(batch_ids):
        folder = (root / 'captures' / batch_id).resolve()
        if folder.parent != (root / 'captures').resolve():
            continue
        manifest_path = folder / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path, {})
        saved = []
        for item in manifest.get('samples', []):
            image = (folder / item['image']).resolve()
            key = (Path('captures') / batch_id / item['image']).as_posix()
            if key in labels or image.parent != folder:
                saved.append(item)
                continue
            image.unlink(missing_ok=True)
            removed += 1
        if saved:
            manifest['samples'] = saved
            write_json(manifest_path, manifest)
        else:
            shutil.rmtree(folder)
    return removed


def clean_all_unlabelled_samples(root):
    """Prune every unlabelled image in one template and keep reviewed samples."""
    root = Path(root).resolve()
    sessions = capture_sessions(root)
    removed = clean_unlabelled_capture_batches(root, [session['id'] for session in sessions])
    existing = {session['id'] for session in capture_sessions(root)}
    hidden_path = root / 'hidden_batches.json'
    hidden = [batch_id for batch_id in read_json(hidden_path, []) if batch_id in existing]
    if hidden_path.exists() or hidden:
        write_json(hidden_path, hidden)
    return removed


class DatasetImportWorker(QThread):
    completed = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, source, destination):
        super().__init__()
        self.source, self.destination = source, destination

    def run(self):
        try:
            self.completed.emit(import_dataset(self.source, self.destination))
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self):
        # Copying keeps existing data intact; finish the active import before closing.
        pass


def read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise IOError('无法编码样本图片')
    encoded.tofile(str(path))


def crop_digits(frame, roi, rotation=0):
    return crop_roi(frame, roi, rotation, strict=True)


def read_frame_at(cap, seconds):
    """Decode to the requested timestamp; reuse the cursor for dense sampling.

    Some recordings seek to an earlier frame (e.g. request 5s, get 4.1s).
    Reading forward also avoids saving the same keyframe on every 0.2s request.
    """
    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if not (position > 0 and timestamp < seconds and seconds - timestamp <= 5):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds-1.0)*1000)
    while True:
        ok, frame = cap.read()
        if not ok:
            return False, None
        if cap.get(cv2.CAP_PROP_POS_MSEC)/1000 >= seconds-1e-6:
            return True, frame


def capture_sessions(root):
    sessions = []
    for manifest in Path(root).glob('captures/*/manifest.json'):
        session = read_json(manifest, {})
        session['id'] = manifest.parent.name
        session['created_at'] = session.get('created_at', manifest.stat().st_mtime)
        sessions.append(session)
    return sorted(sessions, key=lambda s: s['created_at'], reverse=True)


def samples(root, batch=None):
    root = Path(root)
    rows = []
    for session in capture_sessions(root):
        if batch and session['id'] != batch:
            continue
        for item in session.get('samples', []):
            row = dict(item, source=session['source'], unit=session['unit'], batch=session['id'])
            row['key'] = (Path('captures') / session['id'] / item['image']).as_posix()
            rows.append(row)
    return rows


def set_label(root, key, label):
    if label is not None and label != 'invalid' and (type(label) is not int or not 0 <= label <= 100):
        raise ValueError('标签必须是 0～100，或不可辨认')
    path = Path(root) / 'labels.json'
    labels = read_json(path, {})
    if label is None:
        labels.pop(key, None)
    else:
        labels[key] = label
    write_json(path, labels)


def validate_ignore_regions(regions):
    if not isinstance(regions, (list, tuple)) or len(regions) > 32:
        raise ValueError('忽略区域格式无效（最多 32 个）')
    result = []
    for rect in regions:
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError('忽略区域必须包含四个坐标')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in rect):
            raise ValueError('忽略区域坐标无效')
        x1, y1, x2, y2 = rect
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('忽略区域超出图片范围')
        result.append(tuple(float(v) for v in rect))
    return tuple(result)


def load_ignore_regions(root):
    path = Path(root) / 'recognition.json'
    if not path.exists():
        return ()
    try:
        settings = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'无法读取模板忽略区域：{path}') from exc
    if not isinstance(settings, dict) or settings.get('version') != 1:
        raise ValueError('模板忽略区域版本无效')
    return validate_ignore_regions(settings.get('ignore_regions', []))


def save_ignore_regions(root, regions):
    write_json(Path(root) / 'recognition.json',
               {'version': 1, 'ignore_regions': validate_ignore_regions(regions)})


def ignored_pixels(shape, regions):
    h, w = shape[:2]
    ignored = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in regions:
        ignored[math.floor(y1*h):math.ceil(y2*h),
                math.floor(x1*w):math.ceil(x2*w)] = True
    return ignored


def mask_ocr_regions(image, regions=()):
    """Remove excluded icons before OCR, preserving either light or dark backgrounds."""
    if image is None or not image.size or not regions:
        return image
    ignored = ignored_pixels(image.shape, regions)
    if ignored.all():
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # The dominant intensity bin estimates the screen background while excluding
    # icons from estimation. A light LCD must not receive a black rectangle.
    visible = gray[~ignored]
    bins = visible // 16
    dominant = int(np.argmax(np.bincount(bins, minlength=16)))
    background = (~ignored) & (gray // 16 == dominant)
    color = np.median(image[background], axis=0).astype(image.dtype)
    result = image.copy()
    result[ignored] = color
    return result


def feature(image, ignore_regions=()):
    """Align the tall digit band, excluding short icons and empty background."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ignored = ignored_pixels(gray.shape, ignore_regions) if ignore_regions else None
    values = gray[~ignored] if ignored is not None else gray
    if not values.size:
        return None
    lo, hi = np.percentile(values, [10, 99])
    if hi - lo < 25:
        return None
    mask = np.uint8(gray > lo + .50 * (hi - lo)) * 255
    if ignored is not None:
        mask[ignored] = 0
    if not .015 < np.count_nonzero(mask) / values.size < .65:
        return None
    # Closing is for finding the bounds only: keep the original strokes for
    # comparison. The vertical kernel reconnects the two strokes of digit 1.
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    if ignored is not None:
        closed[ignored] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    components = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 5]
    if not components:
        return None
    tallest = max(s[cv2.CC_STAT_HEIGHT] for s in components)
    if tallest < 8:
        return None
    digits = [s for s in components if s[cv2.CC_STAT_HEIGHT] >= tallest*.65]
    x1 = max(0, min(s[0] for s in digits)-2)
    y1 = max(0, min(s[1] for s in digits)-2)
    x2 = min(mask.shape[1], max(s[0]+s[2] for s in digits)+2)
    y2 = min(mask.shape[0], max(s[1]+s[3] for s in digits)+2)
    mask = mask[y1:y2, x1:x2]
    # Keep aspect ratio; black padding makes different ROI sizes comparable.
    h, w = mask.shape
    scale = min(96 / w, 64 / h)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((72, 104), np.uint8)
    x, y = (104-width)//2, (72-height)//2
    canvas[y:y+height, x:x+width] = small
    return canvas


class SegmentMatcher:
    def __init__(self, root, threshold=.88, margin=.015):
        self.roots = [Path(p) for p in library_paths(root)]
        self.root = self.roots[0] if self.roots else None
        self.threshold, self.margin = threshold, margin
        self.templates = []
        self.template_regions = []
        self.last_reason = '尚未匹配'
        seen = set()
        for library in self.roots:
            regions = load_ignore_regions(library)
            for key, label in read_json(library / 'labels.json', {}).items():
                path = (library / key).resolve()
                if not path.is_relative_to(library.resolve()) or not path.is_file():
                    continue
                if label != 'invalid' and (type(label) is not int or not 0 <= label <= 100):
                    continue
                img = feature(read_image(path), regions)
                if img is not None and (label, regions, img.tobytes()) not in seen:
                    seen.add((label, regions, img.tobytes()))
                    self.templates.append((label, img))
                    self.template_regions.append(regions)
        self.template_ink = [float(template.sum()) for _, template in self.templates]
        self.labels = {label for label, _ in self.templates if type(label) is int}

    def extract_soc(self, image):
        candidates = {regions: feature(image, regions) for regions in set(self.template_regions)}
        if not self.labels or all(c is None for c in candidates.values()):
            self.last_reason = '数字画面不完整或模板没有有效标签'
            return None, 0.0
        # Small translations tolerated, but ambiguous digits are rejected.
        prepared = {regions: (cv2.copyMakeBorder(c, 3, 3, 3, 3, cv2.BORDER_CONSTANT), float(c.sum()))
                    for regions, c in candidates.items() if c is not None}
        scores = {}
        for (label, template), template_ink, regions in zip(self.templates, self.template_ink, self.template_regions):
            if regions not in prepared:
                continue
            padded, candidate_ink = prepared[regions]
            correlation = float(cv2.minMaxLoc(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED))[1])
            # An icon-only blink may correlate with a digit. Penalize missing
            # illuminated strokes, while tolerating brighter/thicker real digits.
            score = correlation * min(1.0, candidate_ink / template_ink)
            scores[label] = max(scores.get(label, -1), score)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        label, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1
        if label == 'invalid':
            self.last_reason = '匹配到不可辨认样本'
        elif score < self.threshold:
            self.last_reason = f'相似度不足：{score:.3f} < {self.threshold:.3f}'
        elif len(ranked) > 1 and score - runner_up < (.005 if ranked[1][0] == 'invalid' else self.margin):
            self.last_reason = f'候选难以区分：{label}% / {ranked[1][0]}，差值 {score-runner_up:.3f}'
        else:
            self.last_reason = f'匹配 {label}%，相似度 {score:.3f}'
            return label, score
        return None, max(0.0, score)


class SampleCaptureWorker(QThread):
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(str, int, bool)
    failed = pyqtSignal(str)

    def __init__(self, source, root, roi, rotation=0, unit=1, interval=5.0,
                 start=0.0, end=None):
        super().__init__()
        self.source, self.root = str(source), Path(root)
        self.roi, self.rotation, self.unit = tuple(roi), rotation, unit
        self.interval, self.start_sec, self.end_sec = max(.2, float(interval)), float(start), end
        self.cancelled = False
        self.batch_id = None

    def stop(self):
        self.cancelled = True

    def run(self):
        cap = None
        manifest = None
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise ValueError('无法打开视频：' + self.source)
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count <= 0:
                raise ValueError('视频帧率或长度无效')
            duration = count / fps
            end = min(duration, float(self.end_sec)) if self.end_sec is not None else duration
            if not 0 <= self.start_sec < end:
                raise ValueError('采集结束时间必须大于开始时间')
            folder = self.root / 'captures' / uuid4().hex
            self.batch_id = folder.name
            folder.mkdir(parents=True, exist_ok=False)
            manifest = folder / 'manifest.json'
            session = dict(source=self.source, unit=self.unit, roi=self.roi, created_at=time.time(),
                           rotation=self.rotation, fps=fps, interval=self.interval,
                           start=self.start_sec, end=end, status='collecting', samples=[])
            write_json(manifest, session)
            total = int(np.ceil((end-self.start_sec) / self.interval))
            for i in range(total):
                if self.cancelled:
                    break
                requested = self.start_sec + i*self.interval
                # Seek by timestamp and retain the actual decoded frame/time.
                ok, frame = read_frame_at(cap, requested)
                if not ok:
                    # Report ordinary EOF only when a preceding frame proves
                    # we actually reached the tail, not for an interior error.
                    previous = session['samples'][-1] if session['samples'] else None
                    if (previous and requested >= duration-max(.25, 2/fps)
                            and previous['frame_index'] >= count-max(10, int(fps*.5))):
                        session['end_reason'] = 'video_eof'
                        break
                    raise IOError(f'视频在 {requested:.2f} 秒读取失败；之前的样本已保存')
                image = crop_digits(frame, self.roi, self.rotation)
                name = f'{i:07d}.png'
                write_image(folder / name, image)
                session['samples'].append(dict(image=name, requested_sec=requested,
                    video_sec=cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                    frame_index=max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1)))
                if i % 50 == 0:
                    write_json(manifest, session)
                self.progress.emit(i+1, total)
            session['status'] = 'cancelled' if self.cancelled else 'complete'
            write_json(manifest, session)
            self.completed.emit(str(self.root), len(session['samples']), self.cancelled)
        except Exception as exc:
            if manifest is not None:
                session['status'] = 'error'
                session['error'] = str(exc)
                write_json(manifest, session)
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
