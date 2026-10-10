from src.detection.postprocessing import Detection
from src.storage.repository import WildlifeRepository


def test_duplicate_detection_prevention(tmp_path):
    db = tmp_path / "test.db"
    repo = WildlifeRepository(db)
    job_id = repo.create_job("a.jpg", "image", "yolo11n.pt")
    d = Detection(0, "bear", "bear", 0.9, 1, 2, 3, 4, frame_index=0)
    n1 = repo.insert_detections(job_id, "a.jpg", "image", [d], None, 5.0, "yolo11n.pt")
    n2 = repo.insert_detections(job_id, "a.jpg", "image", [d], None, 5.0, "yolo11n.pt")
    assert n1 == 1
    assert n2 == 0


def test_job_hash_dedupe(tmp_path):
    db = tmp_path / "test.db"
    repo = WildlifeRepository(db)
    j1 = repo.create_job("x.jpg", "batch_image", "w.pt", input_hash="abc")
    repo.finish_job(j1, "completed")
    assert repo.job_exists_for_hash("abc") == j1
