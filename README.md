## Algorithm
- Query images from the dataset, giving a score
- Classify and query images, giving label and scores
## dataset: 
**for classification**

-root

  -- train: img1,img2,img3, ...
  
  -- test: img1,img2,img3, ...

**for query**

-root <br>
  --train <br>
       --- CLASS 1: img1,img2,img3, ... <br>
       --- CLASS 2: img1,img2,img3, ...<br>
       ... <br>
   --test - <br>
## How to run?
**for classification**
```
python findFeatures.py -t /media/ubuntu/zoro/ubuntu/data/train/00/image_0
python query.py -i /media/ubuntu/zoro/ubuntu/data/train/00/image_0/000045.png
```
**for query**

```
python findFeatures.py -t /media/ubuntu/zoro/ubuntu/data/train/station/train
python getClass.py -t dataset/test --visualize
python getClass.py -i /media/ubuntu/zoro/ubuntu/data/train/station/train/s2/001800.png -v
```
