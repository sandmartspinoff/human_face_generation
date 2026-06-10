import os
import cv2
import mediapipe as mp
import json


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict['clips'].items():
        save_name = key+".mp4"
        ytb_id = val['ytb_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']

        bbox = [val['bbox']['top'], val['bbox']['bottom'],
                val['bbox']['left'], val['bbox']['right']]
        yield ytb_id, save_name, time, bbox


def get_top_face_candidates_from_video(
    cap,
    top_k=1,
    sample_every=5,
    min_detection_confidence=0.5,
):
    """
    Returns top-k single-face frames from an opened cv2.VideoCapture.

    Each returned item:
    {
        "frame_idx": int,
        "confidence": float,
        "frame": np.ndarray
    }
    """

    candidates = []

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_detection_confidence,
    ) as face_detection:

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            detections = results.detections

            # reject no-face / multi-face
            if detections is None or len(detections) != 1:
                frame_idx += 1
                continue

            detection = detections[0]
            confidence = float(detection.score[0])

            candidates.append({
                "frame_idx": frame_idx,
                "confidence": confidence,
                "frame": frame.copy(),
            })

            frame_idx += 1

    candidates = sorted(
        candidates,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return candidates[:top_k]


def save_top_frames_for_dataset(
    json_path,
    processed_vid_root,
    processed_img_root,
    top_k=4,
    sample_every=5,
):
    os.makedirs(processed_img_root, exist_ok=True)

    for vid_id, save_vid_name, time, bbox in load_data(json_path):
        processed_vid_path = os.path.join(processed_vid_root, save_vid_name)

        if not os.path.exists(processed_vid_path):
            print(f"missing video, skip: {save_vid_name}")
            continue

        video_stem = os.path.splitext(save_vid_name)[0]

        cap = cv2.VideoCapture(processed_vid_path)

        if not cap.isOpened():
            print(f"failed to open: {processed_vid_path}")
            continue

        top_candidates = get_top_face_candidates_from_video(
            cap,
            top_k=top_k,
            sample_every=sample_every,
        )

        cap.release()

        if len(top_candidates) == 0:
            print(f"no valid single-face frames: {save_vid_name}")
            continue

        for rank, item in enumerate(top_candidates, start=1):
            frame = item["frame"]
            frame_idx = item["frame_idx"]
            confidence = item["confidence"]

            img_save_path = os.path.join(
                processed_img_root,
                f"{video_stem}_top{rank}_frame{frame_idx}_conf{confidence:.3f}.jpg",
            )

            cv2.imwrite(img_save_path, frame)

        print(f"saved {len(top_candidates)} frames: {save_vid_name}")

if __name__ == "__main__":
    save_top_frames_for_dataset("/workspace/data/celebvhq_info.json", 
    "/workspace/data/35666", 
    "/workspace/data/processed-celebvhq-available")