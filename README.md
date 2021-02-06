# pytorch_tutorial
- pytorch를 다시 공부하면서 직접 코드를 만들어 보고 있습니다.

## 01_dataloader 
- HW : notMNIST data 를 불러오는 custom dataset class 를 구현했습니다.
  - download.ipynb : notMNIST_small dataset을 다운로드 받고 압축해제를 합니다. 추가적으로 notMNIST.py의 코드도 붙여넣었습니다.
  - notMNIST.py : torch.utils.data.Dataset class를 상속받아 map-style dataset을 구현했습니다. os, glob, PIL 라이브러리의 사용을 익숙하도록 해야겠습니다.
    - 데이터셋을 다운로드하는 부분을 추가해야 합니다.
