import numpy as np

def A1_A0239038B(X: np.ndarray, A: np.ndarray, y: np.ndarray):
    InvXTAX = np.linalg.pinv(X.T @ A @ X)
    w = InvXTAX @ X.T @ A @ y
    return InvXTAX, w