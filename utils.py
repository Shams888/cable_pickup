"""Small helpers for model inference."""


def filter_result_by_box_size(result, min_width, logger=None):
    """
    Drop detections whose box width is below the given threshold.
    """
    n_before = 0 if result.boxes is None else len(result.boxes)

    # Drop detections whose box is smaller than the width threshold.
    if n_before > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        keep = [
            i for i, (x1, y1, x2, y2) in enumerate(xyxy)
            if (x2 - x1) >= min_width
        ]
        n_dropped = n_before - len(keep)
        if n_dropped and logger is not None:
            logger.info(
                f"infer: dropped {n_dropped}/{n_before} mask(s) "
                f"(need w>={min_width})"
            )
        result = result[keep] if keep else result[[]]

    return result
