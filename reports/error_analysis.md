# Error analysis on the validation halves

54042 evaluated ground-truth boxes (second half of the seven MOT17 training sequences), matched to the tuned trackers with the CLEAR MOT rule at IoU 0.5. `dets >= t` is the share of boxes covered by a YOLO11x detection with score >= t (IoU >= 0.5), an upper bound on what a tracker using those boxes can recall. `IDSW/1k` is ID switches per 1000 boxes in the bin.

| tracker | recall | ID switches |
|---|---|---|
| SORT, tuned | 60.4 % | 512 |
| ByteTrack, tuned | 61.1 % | 269 |

## By visibility (annotated fraction of the person that is not occluded)

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 11576 | 19.9 | 20.2 | 38.0 | 41.8 | 44.8 | 13.6 | 9.1 |
| 0.2-0.4 | 8584 | 27.0 | 30.7 | 37.0 | 43.7 | 48.1 | 8.4 | 5.0 |
| 0.4-0.6 | 6951 | 53.6 | 55.3 | 58.9 | 66.0 | 68.3 | 11.1 | 5.8 |
| 0.6-0.8 | 5970 | 78.4 | 80.1 | 83.5 | 89.4 | 91.6 | 11.6 | 3.4 |
| 0.8-1.0 | 20961 | 93.5 | 92.8 | 95.6 | 96.7 | 97.2 | 6.5 | 2.9 |

## By box height

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| 0-50 px | 1636 | 30.7 | 31.2 | 39.7 | 44.3 | 47.4 | 22.6 | 9.2 |
| 50-100 px | 7613 | 47.2 | 47.0 | 57.8 | 62.1 | 65.9 | 17.6 | 9.1 |
| 100-200 px | 27204 | 55.4 | 56.6 | 63.8 | 69.5 | 71.9 | 6.9 | 3.3 |
| >= 200 px | 17589 | 76.7 | 77.0 | 81.3 | 82.1 | 82.7 | 8.7 | 5.3 |

## By overlap with another person (max IoU with any other box in the frame)

| bin | boxes | recall SORT, tuned | recall ByteTrack, tuned | recall dets >= 0.2 | recall dets >= 0.1 | recall any det | IDSW/1k SORT, tuned | IDSW/1k ByteTrack, tuned |
|---|---|---|---|---|---|---|---|---|
| none | 12976 | 66.1 | 65.9 | 67.6 | 70.8 | 72.2 | 4.1 | 1.1 |
| 0-0.1 | 10007 | 66.4 | 66.7 | 68.7 | 72.0 | 73.8 | 8.7 | 4.5 |
| 0.1-0.3 | 14206 | 65.2 | 65.8 | 67.3 | 70.6 | 72.8 | 12.2 | 6.6 |
| > 0.3 | 16853 | 48.4 | 50.2 | 68.2 | 73.4 | 76.1 | 11.7 | 6.9 |

## ID switches by frames unmatched before the switch

Gap 0 means the person was matched in the previous frame and the ID jumped to another track (usually a swap between neighbours). Longer gaps are identities lost while the person was missed, typically during an occlusion.

| tracker | gap 0 | gap 1-10 | gap 11-30 | gap 31+ |
|---|---|---|---|---|
| SORT, tuned | 161 | 193 | 102 | 56 |
| ByteTrack, tuned | 68 | 88 | 71 | 42 |

## Per sequence

| sequence | camera | boxes | mean overlap | share vis < 0.4 | IDSW SORT, tuned | IDSW ByteTrack, tuned |
|---|---|---|---|---|---|---|
| MOT17-02 | static | 9913 | 0.318 | 60 % | 106 | 58 |
| MOT17-04 | static | 24226 | 0.198 | 35 % | 71 | 32 |
| MOT17-05 | moving | 3363 | 0.178 | 37 % | 96 | 56 |
| MOT17-09 | static | 2892 | 0.351 | 39 % | 31 | 23 |
| MOT17-10 | moving | 5946 | 0.155 | 21 % | 113 | 58 |
| MOT17-11 | moving | 4527 | 0.175 | 31 % | 33 | 20 |
| MOT17-13 | moving | 3175 | 0.148 | 18 % | 62 | 22 |
