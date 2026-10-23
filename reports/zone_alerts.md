# Danger-zone alerts per identity (static-camera sequences, validation halves)

Zone: image fractions x 0.3-0.7, y 0.55-1.0 (lower middle). An alert fires once per visit when a person's feet stay in the zone for 1 s; gaps under 0.5 s do not end a visit. `visits` is the number of alerts the ground truth produces. Tracker alerts are assigned to the person their track covers: `duplicate` alerts are the same person alerted again after an ID switch, `false` alerts come from a track not matched to an annotated pedestrian in that frame (a false track, a distractor such as a static person, or a box off by more than IoU 0.5), and `missed` visits got no alert. With under 30 visits per tracker, differences of a few alerts are within noise.

| tracker | sequence | visits | alerts | correct | duplicate | false | missed |
|---|---|---|---|---|---|---|---|
| SORT, tuned | MOT17-02 | 10 | 12 | 8 | 2 | 2 | 2 |
| SORT, tuned | MOT17-04 | 8 | 14 | 7 | 3 | 4 | 1 |
| SORT, tuned | MOT17-09 | 9 | 8 | 7 | 0 | 1 | 2 |
| SORT, tuned | total | 27 | 34 | 22 | 5 | 7 | 5 |
| ByteTrack, tuned | MOT17-02 | 10 | 12 | 7 | 1 | 4 | 3 |
| ByteTrack, tuned | MOT17-04 | 8 | 10 | 7 | 2 | 1 | 1 |
| ByteTrack, tuned | MOT17-09 | 9 | 12 | 7 | 0 | 5 | 2 |
| ByteTrack, tuned | total | 27 | 34 | 21 | 3 | 10 | 6 |
