# Error analysis on the validation halves

54042 evaluated ground-truth boxes (second half of the seven MOT17 training sequences), matched to the tuned trackers with the CLEAR MOT rule at IoU 0.5. `dets >= t` is the share of boxes covered by a YOLO11x detection with score >= t (IoU >= 0.5), an upper bound on what a tracker using those boxes can recall. `IDSW/1k` is ID switches per 1000 boxes in the bin.

| tracker | recall | ID switches |
|---|---|---|
| SORT, tuned | 60.5 % | 518 |
| ByteTrack, tuned | 61.2 % | 271 |

## By visibility (annotated fraction of the person that is not occluded)

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 11576 | 20.0 | 20.2 | 38.0 | 41.8 | 44.8 | 13.7 | 9.1 |
| 0.2-0.4 | 8584 | 27.1 | 30.8 | 37.0 | 43.7 | 48.1 | 8.6 | 5.2 |
| 0.4-0.6 | 6951 | 53.7 | 55.4 | 58.9 | 66.0 | 68.3 | 11.4 | 5.8 |
| 0.6-0.8 | 5970 | 78.6 | 80.3 | 83.5 | 89.4 | 91.6 | 11.6 | 3.4 |
| 0.8-1.0 | 20961 | 93.5 | 92.8 | 95.6 | 96.7 | 97.2 | 6.5 | 2.9 |

## By box height

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| 0-50 px | 1636 | 30.7 | 31.2 | 39.7 | 44.3 | 47.4 | 22.6 | 9.2 |
| 50-100 px | 7613 | 47.4 | 47.3 | 57.8 | 62.1 | 65.9 | 18.0 | 9.3 |
| 100-200 px | 27204 | 55.4 | 56.7 | 63.8 | 69.5 | 71.9 | 7.0 | 3.3 |
| >= 200 px | 17589 | 76.7 | 77.0 | 81.3 | 82.1 | 82.7 | 8.7 | 5.3 |

## By overlap with another person (max IoU with any other box in the frame)

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| none | 12976 | 66.1 | 65.9 | 67.6 | 70.8 | 72.2 | 4.2 | 1.2 |
| 0-0.1 | 10007 | 66.4 | 66.7 | 68.7 | 72.0 | 73.8 | 8.7 | 4.5 |
| 0.1-0.3 | 14206 | 65.3 | 66.0 | 67.3 | 70.6 | 72.8 | 12.4 | 6.6 |
| > 0.3 | 16853 | 48.5 | 50.3 | 68.2 | 73.4 | 76.1 | 11.9 | 6.9 |

## ID switches by frames unmatched before the switch

Gap 0 means the person was matched in the previous frame and the ID jumped to another track (usually a swap between neighbours). Longer gaps are identities lost while the person was missed, typically during an occlusion.

| tracker | gap 0 | gap 1-10 | gap 11-30 | gap 31+ |
|---|---|---|---|---|
| SORT, tuned | 162 | 196 | 105 | 55 |
| ByteTrack, tuned | 69 | 90 | 70 | 42 |

## Per sequence

| sequence | camera | boxes | mean overlap | share vis < 0.4 | IDSW SORT, tuned | IDSW ByteTrack, tuned |
|---|---|---|---|---|---|---|
| MOT17-02 | static | 9913 | 0.318 | 60 % | 111 | 59 |
| MOT17-04 | static | 24226 | 0.198 | 35 % | 72 | 33 |
| MOT17-05 | moving | 3363 | 0.178 | 37 % | 96 | 56 |
| MOT17-09 | static | 2892 | 0.351 | 39 % | 31 | 23 |
| MOT17-10 | moving | 5946 | 0.155 | 21 % | 113 | 58 |
| MOT17-11 | moving | 4527 | 0.175 | 31 % | 33 | 20 |
| MOT17-13 | moving | 3175 | 0.148 | 18 % | 62 | 22 |
