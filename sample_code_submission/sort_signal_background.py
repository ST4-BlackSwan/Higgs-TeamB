def sort(signal_score, signal_background, val_seuil=0.7):
    s = 0
    b = 0
    for i in signal_background:
        if i >= val_seuil:
            b += 1
    for i in signal_score:
        if i >= val_seuil:
            s += 1
    return (s, b)
