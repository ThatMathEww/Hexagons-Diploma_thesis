# Import knihoven
import numpy as np
cimport numpy as np
import cv2
from numpy cimport float32_t

# Deklarace typů pro vstup a výstup
ctypedef float32_t DTYPE_t
ctypedef np.int32_t DTYPE_INT_t

# Deklarace funkce
cpdef np.ndarray[DTYPE_t, ndim=2] calculate_def_roi(np.ndarray[DTYPE_t, ndim=2] p_old,
                                                    np.ndarray[DTYPE_t, ndim=2] p_new,
                                                    np.ndarray[DTYPE_t, ndim=2] roi_points,
                                                    float radius,
                                                    int num_cores):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(num_cores)

    cdef int length_1 = roi_points.shape[0]
    cdef int length_2 = p_old.shape[0]

    # Inicializace výstupního pole
    cdef np.ndarray[DTYPE_t, ndim=2] def_roi = np.empty((length_1, 2), dtype=np.float32)
    # cdef np.ndarray[DTYPE_t, ndim=2] def_roi = np.empty((0, 2), dtype=np.float32)
    # cdef list def_roi = []

    # Proměnné pro výpočet
    cdef int i
    cdef float c
    cdef np.ndarray[DTYPE_INT_t, ndim=2] selected_indices = np.empty((length_1,length_2), dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] distances
    cdef np.ndarray[DTYPE_INT_t, ndim=1] selected_ind

    # Pro každý referenční bod
    for i in range(length_1):
        selected_ind = np.zeros(length_2, dtype=np.int32)
        c = 0.95

        # Výpočet vzdálenosti mezi každým bodem a referenčním bodem
        while selected_ind.sum() < 6:
            c += 0.05
            distances = np.linalg.norm(p_old - roi_points[i], axis=1)
            selected_ind = np.array(distances <= radius * c, dtype=np.int32)
        selected_indices[i] = selected_ind

        # Nalezení transformační matice
        """tran_mat = cv2.findHomography(p_old[selected_ind == 1], p_new[selected_ind == 1], cv2.RANSAC, 5.0)[0]

        # Převod bodu pomocí transformační matice a přidání do výstupního pole
        transformed_point = cv2.perspectiveTransform(roi_points[i].reshape(-1, 1, 2), tran_mat)[0][0]
        def_roi = np.vstack([def_roi, transformed_point])
        #def_roi.append(def_roi)"""
    def_roi = np.array([cv2.perspectiveTransform(
        reference_point.reshape(-1, 1, 2),
        cv2.findHomography(p_old[selected_i == 1], p_new[selected_i == 1], cv2.RANSAC, 5.0)[0])[0][0]
                        for reference_point, selected_i in zip(roi_points, selected_indices)],
                       dtype=np.float32)

    # return np.array(def_roi, dtype=np.float32)
    return def_roi
