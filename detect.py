# ================== SORT TRACKER ==================
import numpy as np
from collections import deque
import cv2 # Pastikan cv2 diimport untuk fungsi terkait gambar

# Import modul YOLOv5 yang diperlukan
# Asumsi struktur direktori YOLOv5 standar Anda:
# yolov5/
# ├── models/
# ├── utils/
# └── detect.py (file ini)
try:
    import torch
    from models.common import DetectMultiBackend
    from utils.dataloaders import LoadImages, LoadStreams
    from utils.general import (
        check_img_size, check_imshow, non_max_suppression, scale_boxes,
        increment_path, LOGGER, colorstr
    )
    from utils.torch_utils import select_device
    from ultralytics.utils.plotting import Annotator, colors
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print("Pastikan Anda menjalankan skrip ini dari direktori root YOLOv5")
    print("atau bahwa variabel PATH Anda menyertakan direktori YOLOv5.")
    print("Anda juga mungkin perlu menginstal persyaratan YOLOv5:")
    print("pip install -r requirements.txt")
    exit() # Keluar jika import gagal

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("Warning: scipy not found. Install it for better tracker assignment (pip install scipy).")
    print("Falling back to greedy assignment (less optimal).")


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        """
        Inisialisasi pelacak dengan bounding box awal.
        bbox: [x1, y1, x2, y2]
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.bbox = bbox # bbox saat ini
        self.hits = 1  # Jumlah frame di mana track ini telah diupdate dengan deteksi
        self.no_losses = 0 # Jumlah frame berturut-turut di mana track ini tidak memiliki deteksi yang cocok
        self.age = 0 # Usia track dalam frame
        self.class_id = None # Menyimpan ID kelas terakhir yang terkait dengan track
        self.visible = True # Menunjukkan apakah track saat ini terlihat/aktif (diupdate di frame saat ini)

    def update(self, bbox, class_id=None):
        """
        Memperbarui status pelacak dengan bounding box deteksi baru.
        bbox: [x1, y1, x2, y2] dari deteksi yang cocok
        class_id: ID kelas dari deteksi yang cocok
        """
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0 # Reset losses karena ada deteksi yang cocok
        self.age = 0 # Reset usia karena track diupdate
        if class_id is not None:
            self.class_id = class_id # Perbarui ID kelas
        self.visible = True # Set ke terlihat karena ada update di frame saat ini

    def predict(self):
        """
        Memprediksi posisi bounding box berikutnya (sederhana, tidak menggunakan Kalman filter).
        Untuk implementasi Kalman filter yang sebenarnya, ini akan lebih kompleks.
        """
        self.age += 1 # Tambah usia pada prediksi
        self.visible = False # Asumsikan tidak terlihat sampai diupdate di frame saat ini
        return self.bbox

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Inisialisasi pelacak SORT.
        max_age: Jumlah maksimum frame di mana track bisa 'hilang' tanpa deteksi yang cocok sebelum dihapus.
        min_hits: Jumlah minimum deteksi yang diperlukan agar track menjadi 'terkonfirmasi'.
        iou_threshold: Ambang batas IoU untuk mencocokkan deteksi dengan track.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] # Daftar objek KalmanBoxTracker yang aktif
        self.frame_count = 0 # Penghitung frame

    def update(self, dets=np.empty((0, 6))): # dets: numpy array dengan format [x1, y1, x2, y2, conf, class_id]
        """
        Memperbarui status pelacak dengan deteksi baru.
        dets: numpy array dari deteksi untuk frame saat ini.
              Setiap baris adalah [x1, y1, x2, y2, conf, class_id].
        """
        self.frame_count += 1
        
        # Prediksi posisi track yang ada dan reset status terlihat
        for trk in self.trackers:
            trk.predict()
            trk.no_losses += 1 # Tambah jumlah frame tanpa deteksi yang cocok

        # Matriks IoU antara deteksi baru dan track yang ada
        iou_matrix = np.zeros((len(dets), len(self.trackers)), dtype=np.float32)
        if len(dets) > 0 and len(self.trackers) > 0:
            for d, det in enumerate(dets):
                for t, trk in enumerate(self.trackers):
                    # Hitung IoU antara bounding box deteksi dan bounding box track
                    # det[0:4] adalah [x1, y1, x2, y2] dari deteksi
                    # trk.bbox adalah [x1, y1, x2, y2] dari track
                    iou_matrix[d, t] = self.iou(det[0:4], trk.bbox)

            # Lakukan pencocokan menggunakan Hungarian algorithm (atau fallback greedy)
            # Tujuan: meminimalkan 1 - IoU (yaitu, memaksimalkan IoU)
            try:
                # Menggunakan scipy.optimize.linear_sum_assignment untuk pencocokan optimal
                # Kita mencari pencocokan dengan IoU tertinggi, jadi kita meminimalkan -IoU.
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)

                matches = []
                unmatched_detections = set(range(len(dets)))
                unmatched_trackers = set(range(len(self.trackers)))

                for i in range(len(row_ind)):
                    det_idx, trk_idx = row_ind[i], col_ind[i]
                    if iou_matrix[det_idx, trk_idx] >= self.iou_threshold:
                        matches.append([det_idx, trk_idx])
                        unmatched_detections.remove(det_idx)
                        unmatched_trackers.remove(trk_idx)
            except NameError: # Fallback jika scipy tidak diinstal
                # Greedy matching: Iterasi dan temukan IoU terbesar yang belum cocok
                matches = []
                unmatched_detections = list(range(len(dets)))
                unmatched_trackers = list(range(len(self.trackers)))
                
                greedy_iou_matrix = iou_matrix.copy()

                while len(unmatched_detections) > 0 and len(unmatched_trackers) > 0:
                    max_iou = -1
                    best_det_idx, best_trk_idx = -1, -1

                    for d_idx in unmatched_detections:
                        for t_idx in unmatched_trackers:
                            if greedy_iou_matrix[d_idx, t_idx] > max_iou:
                                max_iou = greedy_iou_matrix[d_idx, t_idx]
                                best_det_idx = d_idx
                                best_trk_idx = t_idx
                    
                    if max_iou >= self.iou_threshold and best_det_idx != -1:
                        matches.append([best_det_idx, best_trk_idx])
                        unmatched_detections.remove(best_det_idx)
                        unmatched_trackers.remove(best_trk_idx)
                        # Invalidate matched row/column to prevent re-matching
                        greedy_iou_matrix[best_det_idx, :] = -1 # Tandai baris deteksi ini sebagai tidak valid
                        greedy_iou_matrix[:, best_trk_idx] = -1 # Tandai kolom tracker ini sebagai tidak valid
                    else:
                        break # Tidak ada lagi pencocokan yang baik

        else: # Tidak ada deteksi atau tidak ada track, semua dianggap tidak cocok
            unmatched_detections = set(range(len(dets)))
            unmatched_trackers = set(range(len(self.trackers)))
            matches = []

        # Perbarui track yang cocok dengan deteksi
        for det_idx, trk_idx in matches:
            # dets[det_idx, :4] adalah [x1, y1, x2, y2]
            # dets[det_idx, 5] adalah class_id
            self.trackers[trk_idx].update(dets[det_idx, :4], int(dets[det_idx, 5]))

        # Buat track baru untuk deteksi yang tidak cocok
        for det_idx in unmatched_detections:
            new_trk = KalmanBoxTracker(dets[det_idx, :4])
            new_trk.class_id = int(dets[det_idx, 5]) # Tetapkan ID kelas awal
            self.trackers.append(new_trk)

        # Hapus track yang hilang (melebihi max_age)
        self.trackers = [trk for trk in self.trackers if trk.no_losses <= self.max_age]

        # Kembalikan track yang telah 'dikonfirmasi' (cukup banyak hit) DAN saat ini 'terlihat'
        output_trackers = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.visible:
                # Format output: [x1, y1, x2, y2, track_id, class_id]
                output_trackers.append([*trk.bbox, trk.id, trk.class_id])
        
        return np.array(output_trackers) if len(output_trackers) > 0 else np.empty((0, 6))

    def iou(self, bb_test, bb_gt):
        """
        Menghitung Intersection over Union (IoU) antara dua bounding box.
        bb_test: bounding box deteksi [x1, y1, x2, y2]
        bb_gt: bounding box ground truth atau track [x1, y1, x2, y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        
        area_test = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        area_gt = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
        union_area = area_test + area_gt - wh
        
        # Hindari pembagian dengan nol
        o = wh / union_area if union_area > 0 else 0.0
        return o

# ================== YOLOv5 DETECTION ==================
import argparse
import os
import platform
import sys
import time
from pathlib import Path


def run(
    weights='yolov5s.pt',
    source='data/images',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device='',
    view_img=False,
    save_img=False,
    project='runs/detect',
    name='exp',
    exist_ok=False,
    vid_stride=1,
):
    source = str(source)
    # Membuat direktori penyimpanan untuk hasil
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pilih perangkat (CPU atau GPU)
    device = select_device(device)
    # Muat model YOLOv5
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Tentukan apakah sumber adalah webcam atau file/direktori
    is_webcam = source.isnumeric() or source.endswith('.streams')

    # Inisialisasi dataset loader
    if is_webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * len(dataset), [None] * len(dataset)

    # Warmup model (jalankan inferensi dummy untuk optimasi performa)
    model.warmup(imgsz=(1 if pt else len(dataset), 3, *imgsz))

    # Inisialisasi pelacak SORT
    # max_age: berapa lama track tetap hidup setelah tidak terdeteksi
    # min_hits: berapa banyak deteksi berturut-turut untuk mengkonfirmasi track
    # iou_threshold: IoU minimum untuk mencocokkan deteksi ke track
    tracker = Sort(max_age=60, min_hits=5, iou_threshold=0.4) 
    
    # Set ini akan menyimpan ID unik dari objek yang telah dihitung total
    # Ini memastikan setiap objek dihitung hanya sekali sepanjang seluruh sesi.
    total_counted_objects = set()
    
    # Dictionary ini akan menyimpan jumlah deteksi per kelas
    detection_counts_per_class = {}
    
    # Loop melalui setiap frame dari dataset
    for path, im, im0s, vid_cap, s in dataset:
        start_time = time.time() # Mulai hitung waktu untuk FPS
        
        # Preprocessing gambar untuk model YOLOv5
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Lakukan inferensi
        pred = model(im, augment=False)
        # Terapkan Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Proses deteksi untuk setiap gambar dalam batch (biasanya 1 gambar per frame)
        for i, det in enumerate(pred):
            # Dapatkan path dan gambar asli
            if is_webcam:
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s.copy()

            original_frame = im0.copy() # Simpan salinan frame asli
            
            # Buat anotator untuk menggambar bounding box pada frame
            detected_frame_display = im0.copy() 
            annotator = Annotator(detected_frame_display, line_width=2, example=str(names))

            # Jika ada deteksi
            if len(det):
                # Ubah koordinat bounding box dari skala gambar model ke skala gambar asli
                # det[:, :4] adalah tensor torch berisi [x1, y1, x2, y2]
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Konversi tensor deteksi ke NumPy array
                # .cpu() diperlukan jika det di GPU
                # .numpy() mengkonversi tensor ke NumPy array
                # det_numpy akan memiliki format: [x1, y1, x2, y2, conf, class_id]
                det_numpy = det.cpu().numpy()
                
                # Perbarui pelacak SORT dengan deteksi NumPy
                tracks = tracker.update(det_numpy)

                # Iterasi melalui track yang dikembalikan oleh SORT
                for track in tracks:
                    x1, y1, x2, y2, track_id, class_id = track
                    track_id = int(track_id)
                    class_id = int(class_id)

                    class_name = names[class_id] # Dapatkan nama kelas
                    
                    # LOGIKA PENGHITUNGAN AKURAT:
                    # Objek hanya dihitung sekali jika track_id-nya belum ada di total_counted_objects
                    # DAN track tersebut telah memenuhi kriteria min_hits (dikonfirmasi).
                    if track_id not in total_counted_objects:
                        # Cari objek tracker yang sesuai untuk memeriksa hits
                        current_track_obj = next((trk for trk in tracker.trackers if trk.id == track_id), None)
                        if current_track_obj and current_track_obj.hits >= tracker.min_hits:
                            total_counted_objects.add(track_id) # Tambahkan ID ke set yang sudah dihitung
                            detection_counts_per_class[class_name] = detection_counts_per_class.get(class_name, 0) + 1


                    # Gambar bounding box hanya untuk track yang saat ini 'visible' (yaitu, ada deteksi di frame ini)
                    current_track = next((trk for trk in tracker.trackers if trk.id == track_id), None)
                    if current_track and current_track.visible:
                        label = f'{class_name} ID-{track_id}'
                        annotator.box_label([x1, y1, x2, y2], label, color=colors(class_id, True))

            # Dapatkan frame yang telah dianotasi
            detected_frame = annotator.result()

            # Atur skala ulang untuk tampilan
            resize_scale = 0.5
            original_frame_resized = cv2.resize(original_frame, (int(original_frame.shape[1] * resize_scale), int(original_frame.shape[0] * resize_scale)))
            detected_frame_resized = cv2.resize(detected_frame, (int(detected_frame.shape[1] * resize_scale), int(detected_frame.shape[0] * resize_scale)))

            # Pengaturan font untuk label
            label_font = cv2.FONT_HERSHEY_SIMPLEX
            label_scale = 0.8
            label_color = (255, 255, 255) # Putih
            label_thickness = 2

            # Tambahkan label 'Original' dan 'Detected'
            cv2.putText(original_frame_resized, 'Original', (10, 30), label_font, label_scale, label_color, label_thickness, cv2.LINE_AA)
            cv2.putText(detected_frame_resized, 'Detected', (10, 30), label_font, label_scale, label_color, label_thickness, cv2.LINE_AA)

            # Gabungkan kedua frame secara horizontal
            combined_frame = cv2.hconcat([original_frame_resized, detected_frame_resized])

            # Hitung dan tampilkan FPS
            fps = 1.0 / (time.time() - start_time)
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(combined_frame, fps_text, (10, combined_frame.shape[0] - 10), label_font, 0.7, (0, 255, 0), 2) # Hijau

            # Tampilkan frame gabungan jika view_img True
            if view_img:
                window_name = 'Dual View Detection'
                if platform.system() == 'Linux':
                    # Penyesuaian ukuran jendela untuk Linux (opsional, untuk tampilan yang lebih baik)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(window_name, combined_frame)
                # Tunggu 1ms dan periksa penekanan tombol 'q'
                if cv2.waitKey(1) == ord('q'):
                    break

            # Simpan hasil jika save_img True
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir / Path(p).name), combined_frame)
                else: # Mode video atau stream
                    if vid_path[i] != str(save_dir / Path(p).name):
                        vid_path[i] = str(save_dir / Path(p).name)
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release() # Tutup writer video sebelumnya jika ada

                        if vid_cap: # Ambil properti video asli
                            fps_v = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = combined_frame.shape[1]
                            h = combined_frame.shape[0]
                        else: # Atur default jika tidak ada properti video
                            fps_v, w, h = 30, combined_frame.shape[1], combined_frame.shape[0]
                        
                        # Simpan video sebagai MP4
                        save_path_video = str(Path(vid_path[i]).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (w, h))
                    vid_writer[i].write(combined_frame)

    # Cetak hasil jumlah deteksi setelah semua frame diproses
    print("\nHasil jumlah deteksi:")
    for class_name, count in detection_counts_per_class.items():
        print(f"{class_name}: {count}")

    # ========== ❗ TAMBAHAN AGAR JENDELA TIDAK LANGSUNG TERTUTUP ❗ ==========
    if view_img:
        print("\nTekan tombol 'q' atau close window untuk keluar...")
        while True:
            # Tetap tampilkan jendela hingga 'q' ditekan atau jendela ditutup secara manual
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Dual View Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows() # Tutup semua jendela OpenCV

def parse_opt():
    # Parsing argumen baris perintah
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='jalur ke model weights')
    parser.add_argument('--source', type=str, default='data/images', help='file/direktori/URL/glob, 0 untuk webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='ukuran inferensi h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='ambang batas kepercayaan deteksi')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='ambang batas IoU NMS')
    parser.add_argument('--device', default='', help='perangkat cuda, misal 0 atau cpu')
    parser.add_argument('--view-img', action='store_true', help='tampilkan hasil di jendela')
    parser.add_argument('--save-img', action='store_true', help='simpan hasil gambar/video')
    parser.add_argument('--project', default='runs/detect', help='simpan ke project/name')
    parser.add_argument('--name', default='exp', help='simpan ke project/name')
    parser.add_argument('--exist-ok', action='store_true', help='project/name yang ada boleh, jangan increment')
    parser.add_argument('--vid-stride', type=int, default=1, help='langkah frame-rate video')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1 # Sesuaikan ukuran gambar jika hanya satu dimensi yang diberikan
    return opt

def main(opt):
    run(**vars(opt)) # Jalankan fungsi deteksi utama dengan argumen yang di-parse

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)