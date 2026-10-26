"""MediaPipe's FaceLandmarker.__del__ calls close(), which waits on MediaPipe's own dispatcher; when the garbage
collector runs it in the middle of another test (seen while importing torch) that wait never returns and the run
hangs. Tests leave the landmarkers to process exit instead."""
try:
    from mediapipe.tasks.python.vision import face_landmarker

    face_landmarker.FaceLandmarker.__del__ = lambda self: None
except ImportError:
    pass
